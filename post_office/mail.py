import sys
import logging, io
from functools import partial
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import connection as db_connection
from django.db.models import Q
from django.template import Context, Template
from django.utils import timezone
from datetime import timedelta
from email.utils import make_msgid
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


from google_auth_httplib2 import AuthorizedHttp
from httplib2 import Http

from .connections import connections
from .lockfile import default_lockfile, FileLock, FileLocked
from .logutils import setup_loghandlers
from post_office.models import Email, EmailTemplate, Log, PRIORITY, STATUS
from .settings import (
    get_available_backends, get_batch_size, get_log_level, get_max_retries, get_message_id_enabled,
    get_message_id_fqdn, get_retry_timedelta, get_sending_order, get_threads_per_process
)
from .signals import email_queued
from .utils import (
    create_attachments, get_email_template, parse_emails, parse_priority, split_emails,upload_to_s3
)

from seed.lib.superperms.orgs.models import Organization
from helpdesk.models import Ticket
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
    status = STATUS.queued # None if priority == PRIORITY.now else STATUS.queued

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
            sender = settings.SERVER_EMAIL

    priority = parse_priority(priority)

    if log_level is None:
        log_level = get_log_level()

    if not commit:
        # if priority == PRIORITY.now:
        #     raise ValueError("send_many() can't be used with priority = 'now'")
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

    """
    if priority == PRIORITY.now:
        # todo disabling this for now
        # email.dispatch(log_level=log_level)
        pass
    """
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


def get_queued(organization=None, extra_q=None):
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
    if extra_q:
        query = query & extra_q

    return Email.objects.filter(query) \
                .select_related('template') \
                .order_by(*get_sending_order()).prefetch_related('attachments')

def get_queued_for_google(org):
    """
    Google has a 2000 emails/day limit, but we don't want huge amounts of bulk emails to prevent other kinds of mail
    from being sent. This method caps the amount of bulk mail and allows other kinds of mail to always be prioritized.
    """
    stagger = False # todo
    gmail_limit = 2000 - 1  # https://support.google.com/a/answer/166852?hl=en&fl=1&sjid=577582152286187260-NC
    priority_min = 300
    priority_query = Q(priority__in=[PRIORITY.now, PRIORITY.high])  # todo or scheduled_time = now
    log_priority_query = Q(email__priority__in=[PRIORITY.now, PRIORITY.high])

    regular_limit = gmail_limit - priority_min
    batch = get_batch_size()
    date = timezone.now() - timedelta(days=1)

    daily_sent = Log.objects.filter(
        status=STATUS.sent,
        date__gte=date,
        email__organization=org
    )
    daily_left = gmail_limit - daily_sent.count()
    logger.info(f'Emails sent today: {daily_sent.count()}')

    # First, prioritize these emails and send as many of them as you can fit in the batch
    priority_queued = get_queued(org, priority_query)
    if daily_left < batch:
        batch = daily_left
    queued_list = priority_queued[:batch]
    queued_count = queued_list.count()
    logger.info(f'Sending {queued_count} priority emails')

    # Next, check if there's any space left in the batch - if not, skip sending the other kinds of emails
    batch_left = batch - queued_count
    if batch_left == 0:
        return Email.objects.filter(id__in=queued_list)

    daily_priority_sent = daily_sent.filter(log_priority_query).count()
    daily_regular_sent = daily_sent.filter(~log_priority_query).count()

    # Get the rest of the emails
    regular_queued = get_queued(org, ~priority_query)
    # If more priority mail has been sent than the min number of priority emails, then we'll adjust the limit for regular mail
    if daily_priority_sent + queued_count > priority_min:
        regular_limit = gmail_limit - daily_priority_sent - len(queued_list)

    # Based on the new limit for regular mail, how much can we send?
    daily_regular_left = regular_limit - daily_regular_sent
    # fill out the batch, or send up to the regular daily limit, whichever is smaller
    regular_batch = batch_left if batch_left < daily_regular_left else daily_regular_left
    # if stagger is on, send a smaller batch of the regular mail
    if stagger and stagger < regular_batch:
        regular_batch = stagger

    regular_list = regular_queued[:regular_batch]
    queued_list = Email.objects.filter(Q(id__in=queued_list) | Q(id__in=regular_list))
    logger.info(f'Sending {queued_list.count() - queued_count} regular emails')

    return queued_list

def send_queued(processes=1, log_level=None, ignore_slow=False, log_and_upload=False):
    """
    Sends out all queued mails that has scheduled_time less than now or None
    """
    from seed.models import GOOGLE
    total_sent, total_failed, total_requeued, total_email = 0, 0, 0, 0
    gmail_limit = 2000 - 1  # https://support.google.com/a/answer/166852?hl=en&fl=1&sjid=577582152286187260-NC
    date = timezone.now() - timedelta(days=1)
    orgs = Organization.objects.all()
    queued_emails_by_org = {}
    for org in orgs:
        if org.sender and org.sender.auth and org.sender.auth.host_service == GOOGLE:
            queued_unsliced = get_queued_for_google(org)
        else:
            queued_unsliced = get_queued(organization=org)
        queued = []
        if queued_unsliced:
            if org.email_slow_mode and not ignore_slow:
                if queued_unsliced.filter(source_page=Email.HELPDESK).exists():
                    logger.info(f'Sending for {org} in slow mode.')
                    queued += queued_unsliced.exclude(source_page=Email.HELPDESK)[:get_batch_size() - 1]
                    queued += queued_unsliced.filter(source_page=Email.HELPDESK)[:1]
                else:
                    logger.info(f'Turning off slow mode for {org}')
                    queued = queued_unsliced[:get_batch_size()]
                    org.email_slow_mode = False
                    org.save()
            else:
                helpdesk_mail = queued_unsliced.filter(source_page=Email.HELPDESK)
                if ignore_slow or helpdesk_mail.count() < 40:
                    queued = queued_unsliced[:get_batch_size()]
                else:
                    logger.info(f'Turning on slow mode for {org}')
                    # that's a lot of mail. start sending in slow mode
                    org.email_slow_mode = True
                    org.save()

                    # put together list of emails to send slowly
                    queued += queued_unsliced.exclude(source_page=Email.HELPDESK)[:get_batch_size() - 1]
                    queued += helpdesk_mail[:1]

                    # create alert email
                    # todo: add in other users

                    # cc = {'ebeers@clearlyenergy.com', 'vbugnion@clearlyenergy.com'}
                    hd_list_plain = ""
                    hd_list_html = "<ul>\n"
                    for ticket in Ticket.objects.filter(emails__in=helpdesk_mail).distinct():
                        # if ticket.assigned_to:
                        #     cc.add(ticket.assigned_to.email)
                        hd_list_plain += f'- {ticket.emails.filter(status=STATUS.queued).count()} email(s): [{ticket.queue.slug}-{ticket.id}] {ticket.title}\n'
                        hd_list_html += f'<li>{ticket.emails.filter(status=STATUS.queued).count()} email(s): <a href="{ticket.staff_url}">[{ticket.queue.slug}-{ticket.id}] {ticket.title}</a></li>\n'
                    hd_list_html += "</ul>"

                    msg_plain = (
                        'Helpdesk has detected that an unusually large amount of emails have been queued to send, and has automatically turned on "slow mode." '
                        'In slow mode, Helpdesk will send mail at a rate of one email every five minutes. '
                        'Slow mode will automatically turn off once there are no more queued Helpdesk emails.\n\n'
                        'The following tickets currently have queued mail:\n\n'
                        f'{hd_list_plain}\n'
                        "If individual tickets are creating a large volume of mail, the ticket may be causing an email loop, or it may have too many CC'd addresses. "
                        'Here are some solutions:\n\n'
                        '- Do not delete the ticket.\n'
                        "- Go to the ticket's page and toggle the red button at the top, \"Disable Email Notifications.\" "
                            "This will delete any queued emails created by that ticket and prevent them from sending, and it will prevent the ticket from creating and sending out future emails. "
                            "Imports will still be allowed.\n"
                        "- Look at the list of CC'd email addresses, and remove any that are causing issues or should not be on the ticket. "
                            "As new imports come in, you will need to re-check the list of CC'd addresses.\n\n"
                            "If the issue is that too many new tickets are being created, here are some solutions:\n\n"
                        "- Do not delete the ticket.\n"
                        "- Look at the submitter of the tickets, and add their email address to the list of Ignored Addresses. "
                            "If they're already on the list, double-check that both sending and importing are blocked for that address.\n"
                        "- If the tickets are being created by forms, set the forms to staff-only and ensure they're not unlisted.\n\n"
                        "If you take any steps to solve the issue, please reply-all to this email with those steps. "
                        "Alternatively, if this email was a false alarm, please reply-all and let ClearlyEnergy know so that we can manually turn off slow mode."
                    )
                    msg_html = (
                        '<p>Helpdesk has detected that an unusually large amount of emails have been queued to send, and has automatically turned on "slow mode." '
                        'In slow mode, Helpdesk will send mail at a rate of one email every five minutes. '
                        'Slow mode will automatically turn off once there are no more queued Helpdesk emails.</p>'
                        '<p>The following tickets currently have queued mail:</p>'
                        f'{hd_list_html}'
                        "<p>If individual tickets are creating a large volume of mail, the ticket may be causing an email loop, or it may have too many CC'd addresses. "
                        'Here are some solutions:</p>'
                        '<ul>'
                        '<li>Do not delete the ticket.</li>'
                        "<li>Go to the ticket's page and toggle the red button at the top, \"Disable Email Notifications.\" "
                            "This will delete any queued emails created by that ticket and prevent them from sending, and it will prevent the ticket from creating and sending out future emails. "
                            "Imports will still be allowed.</li>"
                        "<li>Look at the list of CC'd email addresses, and remove any that are causing issues or should not be on the ticket. "
                            "As new imports come in, you will need to re-check the list of CC'd addresses.</li>"
                        '</ul>'
                        "<p>If the issue is that too many new tickets are being created, here are some solutions:</p>"
                        '<ul>'
                        "<li>Do not delete the ticket.</li>"
                        "<li>Look at the submitter of the tickets, and add their email address to the list of Ignored Addresses. "
                            "If they're already on the list, double-check that both sending and importing are blocked for that address.</li>"
                        "<li>If the tickets are being created by forms, set the forms to staff-only and ensure they're not unlisted.</li>"
                        '</ul>'
                        "<p>If you take any steps to solve the issue, please reply-all to this email with those steps. "
                        "Alternatively, if this email was a false alarm, please reply-all and let ClearlyEnergy know so that we can manually turn off slow mode.</p>"
                    )
                    # create email object for alert email and add it to the emails that should be sent out right now

                    # _, beam_header = add_custom_header(list(cc))
                    log_mail = send(
                        recipients=['beammonitoring@gmail.com'],
                        sender=org.sender.email_address,
                        subject=f'IMPORTANT: Helpdesk email overflow detected',
                        message=msg_plain,
                        html_message=msg_html,
                        organization=org,
                        source_page=Email.SCRIPT,
                        source_action='email reports',
                        # headers={'X-BEAMHelpdesk-Delivered': beam_header},
                    )
                    queued.append(log_mail)

            queued_emails_by_org[org] = queued
            total_email += len(queued)
        elif org.email_slow_mode:
            org.email_slow_mode = False
            org.save()

    logger.info('Started sending %s emails with %s processes.' % (total_email, processes))

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

    stream = io.StringIO()
    s3_logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    handler = logging.StreamHandler(stream)
    s3_logger.addHandler(handler)
 
    logger.info(
        '%s emails attempted, %s sent, %s failed, %s requeued',
        total_email, total_sent, total_failed, total_requeued,
    )
    if log_and_upload is True:
        s3_logger.info(
        '%s emails attempted, %s sent, %s failed, %s requeued',
        total_email, total_sent, total_failed, total_requeued,)
        upload_to_s3(stream.getvalue(), settings.AWS_STORAGE_BUCKET_NAME, "send_mail.log")
    return total_sent, total_failed, total_requeued


def _send_bulk(emails, uses_multiprocessing=True, log_level=None, organization=None):
    # Multiprocessing does not play well with database connection
    # Fix: Close connections on forking process
    # https://groups.google.com/forum/#!topic/django-users/eCAIY9DAfG0
    # The Google API we use for credentials uses httplib2, which isn't thread-safe
    # Fix: Create an http object for each thread https://googleapis.github.io/google-api-python-client/docs/thread_safety.html
    # Future todo: Update google.oauth2.credentials to googleapiclient, which doesn't use httplib2
    if uses_multiprocessing:
        db_connection.close()

    if log_level is None:
        log_level = get_log_level()

    sent_emails = []
    failed_emails = []  # This is a list of two tuples (email, exception)
    email_count = len(emails)

    logger.info('Process started, sending %s emails' % email_count)

    # set up backend
    sender = organization.sender
    google_credentials = None
    if not sender:
        logger.exception('Emails could not be sent, exiting process')
        return 0, 0, 0
    if sender.auth:
        # todo make sure errors in login() will show up in the logger properly
        connection, error = sender.auth.login(logger=logger, email=sender)
        if error:
            return error
        google_credentials = sender.auth.get_google_credentials()
    else:
        connection = setup_basic_backend(None, sender=sender)

    def send(email):
        http = None
        if google_credentials:
            http = AuthorizedHttp(google_credentials, http=Http())
        try:
            email.dispatch(log_level=log_level, commit=False, disconnect_after_delivery=False, http=http)
            sent_emails.append(email)
            logger.debug('Successfully sent email #%d' % email.id)
        except Exception as e:
            logger.exception('Failed to send email #%d' % email.id)
            failed_emails.append((email, e))

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

def send_queued_mail_until_done(lockfile=default_lockfile, processes=1, log_level=None, ignore_slow=False, log_and_upload=False):
    """
    Send mail in queue batch by batch, until all emails have been processed.
    Updated to only send one batch at a time.
    """
    try:
        with FileLock(lockfile):
            logger.info('Acquired lock for sending queued emails at %s.lock', lockfile)
            # while True:
            try:
                send_queued(processes, log_level, ignore_slow=ignore_slow, log_and_upload=log_and_upload)
            except Exception as e:
                logger.exception(e, extra={'status_code': 500})
                raise

            # Close DB connection to avoid multiprocessing errors
            db_connection.close()

            # if not get_queued().exists():
            #     break
    except FileLocked:
        logger.info('Failed to acquire lock, terminating now.')