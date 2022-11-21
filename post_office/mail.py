import sys
from functools import partial
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import connection as db_connection
from django.db.models import Q
from django.template import Context, Template
from django.utils import timezone
from email.utils import make_msgid
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from O365 import Account

from .connections import connections
from .lockfile import default_lockfile, FileLock, FileLocked
from .logutils import setup_loghandlers
from .models import Email, EmailTemplate, Log, PRIORITY, STATUS
from .settings import (
    get_available_backends, get_batch_size, get_log_level, get_max_retries, get_message_id_enabled,
    get_message_id_fqdn, get_retry_timedelta, get_sending_order, get_threads_per_process,
)
from .signals import email_queued
from .utils import (
    create_attachments, get_email_template, parse_emails, parse_priority, split_emails,
)

from seed.lib.superperms.orgs.models import Organization
from seed.models.email_settings import MICROSOFT, GOOGLE
from seed.utils.email import setup_basic_backend

logger = setup_loghandlers("INFO")


def create(sender, recipients=None, cc=None, bcc=None, subject='', message='',
           html_message='', context=None, scheduled_time=None, expires_at=None, headers=None,
           template=None, priority=None, render_on_delivery=False, commit=True,
           backend='', organization=None, user=None, source_page=None, source_action=None, ticket=None):
    """
    Creates an email from supplied keyword arguments. If template is
    specified, email subject and content will be rendered during delivery.
    """
    priority = parse_priority(priority)
    status = None if priority == PRIORITY.now else STATUS.queued

    if recipients is None:
        recipients = []
    if cc is None:
        cc = []
    if bcc is None:
        bcc = []
    if context is None:
        context = ''
    message_id = make_msgid(domain=get_message_id_fqdn()) if get_message_id_enabled() else None

    # If email is to be rendered during delivery, save all necessary
    # information
    if render_on_delivery:
        email = Email(
            from_email=sender,
            to=recipients,
            cc=cc,
            bcc=bcc,
            scheduled_time=scheduled_time,
            expires_at=expires_at,
            message_id=message_id,
            headers=headers, priority=priority, status=status,
            context=context, template=template, backend_alias=backend,
            organization=organization, user=user, source_page=source_page, source_action=source_action,
            ticket=ticket
        )

    else:
        if template:
            subject = template.subject
            message = template.content
            html_message = template.html_content

        _context = Context(context or {})
        subject = Template(subject).render(_context)
        message = Template(message).render(_context)
        html_message = Template(html_message).render(_context)

        email = Email(
            from_email=sender,
            to=recipients,
            cc=cc,
            bcc=bcc,
            subject=subject,
            message=message,
            html_message=html_message,
            scheduled_time=scheduled_time,
            expires_at=expires_at,
            message_id=message_id,
            headers=headers, priority=priority, status=status,
            backend_alias=backend,
            template=template,
            organization=organization,
            user=user,
            source_page=source_page,
            source_action=source_action,
            ticket=ticket
        )

    if commit:
        email.save()

    return email


def send(recipients=None, sender=None, template=None, context=None, subject='',
         message='', html_message='', scheduled_time=None, expires_at=None, headers=None,
         priority=None, attachments=None, render_on_delivery=False,
         log_level=None, commit=True, cc=None, bcc=None, language='',
         backend='',
         organization=None, user=None, source_page=None, source_action=None,
         ticket=None, property_views=None, taxlot_views=None):
    # todo save senderobj , not from_address
    try:
        recipients = parse_emails(recipients)
    except ValidationError as e:
        raise ValidationError('recipients: %s' % e.message)

    if organization is None:
        raise ValueError("Email must be sent by an organization.")

    try:
        cc = parse_emails(cc)
    except ValidationError as e:
        raise ValidationError('c: %s' % e.message)

    try:
        bcc = parse_emails(bcc)
    except ValidationError as e:
        raise ValidationError('bcc: %s' % e.message)

    if sender is None:
        if getattr(organization, 'sender', False):
            sender = organization.sender.email_address
        else:
            sender = settings.DEFAULT_FROM_EMAIL

    priority = parse_priority(priority)

    if log_level is None:
        log_level = get_log_level()

    if not commit:
        if priority == PRIORITY.now:
            raise ValueError("send_many() can't be used with priority = 'now'")
        if attachments:
            raise ValueError("Can't add attachments with send_many()")
        if property_views or taxlot_views:
            raise ValueError("Can't associate emails with property/taxlot_views with send_many()")

    if template:
        if subject:
            raise ValueError('You can\'t specify both "template" and "subject" arguments')
        if message:
            raise ValueError('You can\'t specify both "template" and "message" arguments')
        if html_message:
            raise ValueError('You can\'t specify both "template" and "html_message" arguments')

        # template can be an EmailTemplate instance or name
        if isinstance(template, EmailTemplate):
            template = template
            # If language is specified, ensure template uses the right language
            if language and template.language != language:
                template = template.translated_templates.get(language=language)
        else:
            template = get_email_template(template, language)

    if backend and backend not in get_available_backends().keys():
        raise ValueError('%s is not a valid backend alias' % backend)

    email = create(sender, recipients, cc, bcc, subject, message, html_message,
                   context, scheduled_time, expires_at, headers, template, priority,
                   render_on_delivery, commit=commit, backend=backend,
                   organization=organization, user=user, source_page=source_page, source_action=source_action,
                   ticket=ticket)

    if attachments:
        attachments = create_attachments(attachments)
        email.attachments.add(*attachments)
    if property_views:
        email.property_views.set(property_views)
    if taxlot_views:
        email.taxlot_views.set(taxlot_views)

    if priority == PRIORITY.now:
        # todo disabling this for now
        # email.dispatch(log_level=log_level)
        pass
    email_queued.send(sender=Email, emails=[email])

    return email


def send_many(kwargs_list):
    """
    Similar to mail.send(), but this function accepts a list of kwargs.
    Internally, it uses Django's bulk_create command for efficiency reasons.
    Currently send_many() can't be used to send emails with priority = 'now'.
    """
    emails = [send(commit=False, **kwargs) for kwargs in kwargs_list]
    if emails:
        Email.objects.bulk_create(emails)
        email_queued.send(sender=Email, emails=emails)


def get_queued(organization=None):
    """
    Returns the queryset of emails eligible for sending – fulfilling these conditions:
     - Status is queued or requeued
     - Has scheduled_time before the current time or is None
     - Has expires_at after the current time or is None
    """
    now = timezone.now()
    query = (
        (Q(status=STATUS.queued) | Q(status=STATUS.requeued)) &
        (Q(scheduled_time__lte=now) | Q(scheduled_time__isnull=True)) &
        (Q(expires_at__gt=now) | Q(expires_at__isnull=True))
    )
    if organization:
        query = query & (Q(organization=organization))

    return Email.objects.filter(query) \
                .select_related('template') \
                .order_by(*get_sending_order()).prefetch_related('attachments')[:get_batch_size()]


def send_queued(processes=1, log_level=None):
    """
    Sends out all queued mails that has scheduled_time less than now or None
    """
    total_sent, total_failed, total_requeued, total_email = 0, 0, 0, 0
    orgs = Organization.objects.all()
    queued_emails_by_org = {}
    for org in orgs:
        queued = get_queued(organization=org)
        if queued:
            queued_emails_by_org[org] = queued
            total_email += len(queued)

    logger.info('Started sending %s emails with %s processes.' %
                (total_email, processes))

    if log_level is None:
        log_level = get_log_level()

    if queued_emails_by_org:
        for org, queued_emails in queued_emails_by_org.items():
            logger.info(f'Started sending {len(queued_emails)} emails for org ({org.id}) {org.name} with {processes} processes.')
            # Don't use more processes than number of emails
            if total_email < processes:
                processes = total_email

            if processes == 1:
                result = _send_bulk(
                    emails=queued_emails,
                    uses_multiprocessing=False,
                    log_level=log_level,
                    organization=org,
                )
                total_sent += result[0]
                total_failed += result[1]
                total_requeued += result[2]

            else:
                email_lists = split_emails(queued_emails, processes)

                pool = Pool(processes)
                results = pool.map(partial(_send_bulk, organization=org), email_lists)
                pool.terminate()

                total_sent += sum(result[0] for result in results)
                total_failed += sum(result[1] for result in results)
                total_requeued += [result[2] for result in results]

    logger.info(
        '%s emails attempted, %s sent, %s failed, %s requeued',
        total_email, total_sent, total_failed, total_requeued,
    )

    return total_sent, total_failed, total_requeued


def _send_bulk(emails, uses_multiprocessing=True, log_level=None, organization=None):
    # Multiprocessing does not play well with database connection
    # Fix: Close connections on forking process
    # https://groups.google.com/forum/#!topic/django-users/eCAIY9DAfG0
    if uses_multiprocessing:
        db_connection.close()

    if log_level is None:
        log_level = get_log_level()

    sent_emails = []
    failed_emails = []  # This is a list of two tuples (email, exception)
    email_count = len(emails)

    logger.info('Process started, sending %s emails' % email_count)

    def send(email):
        try:
            email.dispatch(log_level=log_level, commit=False,
                           disconnect_after_delivery=False)
            sent_emails.append(email)
            logger.debug('Successfully sent email #%d' % email.id)
        except Exception as e:
            logger.exception('Failed to send email #%d' % email.id)
            failed_emails.append((email, e))

    # set up backend
    sender = organization.sender
    connection = None
    if not sender:
        logger.exception('Emails could not be sent, exiting process')
        return 0, 0, 0
    if sender.auth:
        if sender.auth.host_service == MICROSOFT:
            # set up connection with Microsoft
            auth = {
               'client_id': sender.auth.get_client_id(),
               'client_secret': sender.auth.get_client_secret(),
               'tenant_id': sender.auth.get_tenant_id()
            }
            token_backend = sender.auth.get_access_token_backend()
            connection = Account((auth['client_id'], auth['client_secret']), auth_flow_type='credentials',
                                 tenant_id=auth['tenant_id'], token_backend=token_backend)
            if not connection.authenticate():
                logger.exception('Unable to authenticate Microsoft connection.')
                return 0, 0, 0
        elif sender.auth.host_service == GOOGLE:
            logger.exception('Emails could not be sent - Google is not yet supported. Exiting process')
            return 0, 0, 0
    else:
        connection = setup_basic_backend(None, sender=sender)

    # Prepare emails before we send these to threads for sending
    # So we don't need to access the DB from within threads
    for email in emails:
        # Sometimes this can fail, for example when trying to render
        # email from a faulty Django template
        try:
            email.prepare_email_message(connection=connection)
        except Exception as e:
            logger.exception('Failed to prepare email #%d' % email.id)
            failed_emails.append((email, e))

    number_of_threads = min(get_threads_per_process(), email_count)
    pool = ThreadPool(number_of_threads)

    pool.map(send, emails)
    pool.close()
    pool.join()

    connections.close()

    # Update statuses of sent emails
    email_ids = [email.id for email in sent_emails]
    Email.objects.filter(id__in=email_ids).update(status=STATUS.sent)

    # Update statuses and conditionally requeue failed emails
    num_failed, num_requeued = 0, 0
    max_retries = get_max_retries()
    scheduled_time = timezone.now() + get_retry_timedelta()
    emails_failed = [email for email, _ in failed_emails]

    for email in emails_failed:
        if email.number_of_retries is None:
            email.number_of_retries = 0
        if email.number_of_retries < max_retries:
            email.number_of_retries += 1
            email.status = STATUS.requeued
            email.scheduled_time = scheduled_time
            num_requeued += 1
        else:
            email.status = STATUS.failed
            num_failed += 1

    Email.objects.bulk_update(emails_failed, ['status', 'scheduled_time', 'number_of_retries'])

    # If log level is 0, log nothing, 1 logs only sending failures
    # and 2 means log both successes and failures
    if log_level >= 1:

        logs = []
        for (email, exception) in failed_emails:
            logs.append(
                Log(email=email, status=STATUS.failed,
                    message=str(exception),
                    exception_type=type(exception).__name__)
            )

        if logs:
            Log.objects.bulk_create(logs)

    if log_level == 2:

        logs = []
        for email in sent_emails:
            logs.append(Log(email=email, status=STATUS.sent))

        if logs:
            Log.objects.bulk_create(logs)

    logger.info(
        f'Process finished for ({organization.id}) {organization.name}, %s attempted, %s sent, %s failed, %s requeued',
        email_count, len(sent_emails), num_failed, num_requeued,
    )

    return len(sent_emails), num_failed, num_requeued


def send_queued_mail_until_done(lockfile=default_lockfile, processes=1, log_level=None):
    """
    Send mail in queue batch by batch, until all emails have been processed.
    """
    try:
        with FileLock(lockfile):
            logger.info('Acquired lock for sending queued emails at %s.lock', lockfile)
            while True:
                try:
                    send_queued(processes, log_level)
                except Exception as e:
                    logger.exception(e, extra={'status_code': 500})
                    raise

                # Close DB connection to avoid multiprocessing errors
                db_connection.close()

                if not get_queued().exists():
                    break
    except FileLocked:
        logger.info('Failed to acquire lock, terminating now.')