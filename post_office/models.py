import os

import re
from collections import namedtuple
from uuid import uuid4
from email.mime.nonmultipart import MIMENonMultipart

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.db import models
from django.utils.encoding import smart_str
from django.utils.translation import pgettext_lazy, gettext_lazy as _
from django.utils import timezone
from jsonfield import JSONField
from django.core.files.storage import default_storage

from exchangelib import HTMLBody, Message as EWSMessage, ExtendedProperty, FileAttachment

from post_office import cache
from post_office.fields import CommaSeparatedEmailField

from .connections import connections
from .logutils import setup_loghandlers
from .settings import context_field_class, get_log_level, get_template_engine, get_override_recipients
from .validators import validate_email_with_name, validate_template_syntax

from seed.landing.models import SEEDUser as User
from seed.lib.superperms.orgs.models import Organization
from seed.utils.strings import check_if_context_appropriate


logger = setup_loghandlers("INFO")


PRIORITY = namedtuple('PRIORITY', 'low medium high now')._make(range(4))
STATUS = namedtuple('STATUS', 'sent failed queued requeued')._make(range(4))


class BEAMHeader(ExtendedProperty):
    distinguished_property_set_id = "InternetHeaders"
    property_name = "X-BEAMHelpdesk-Delivered"
    property_type = 'String'


class Email(models.Model):
    """
    A model to hold email information.
    """
    # BEAM
    BEAM = 0
    SCRIPT = 1
    LANDING = 2
    POST_OFFICE = 3
    HELPDESK = 4
    EMAIL_SOURCE_CHOICES = [
        (BEAM, 'BEAM'),
        (SCRIPT, 'Script'),
        (LANDING, 'Landing'),
        (POST_OFFICE, 'Post Office'),
        (HELPDESK, 'Helpdesk'),
    ]
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, blank=True, null=True, related_name='emails')
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True, related_name='emails')

    # M2M field used to account for merged and unmerged views
    property_views = models.ManyToManyField('seed.PropertyView', related_name='emails')
    taxlot_views = models.ManyToManyField('seed.TaxLotView', related_name='emails')
    # When tickets are merged, emails are re-associated with the ticket that has been merged-into.
    ticket = models.ForeignKey('helpdesk.Ticket', on_delete=models.SET_NULL, blank=True, null=True, related_name='emails')

    source_page = models.IntegerField(help_text="Where in BEAM was this email sent from?", choices=EMAIL_SOURCE_CHOICES, default=None, blank=True, null=True)
    source_action = models.CharField(help_text="Why was this email sent?", max_length=300, blank=True, null=True)
    # end BEAM

    PRIORITY_CHOICES = [(PRIORITY.low, _("low")), (PRIORITY.medium, _("medium")),
                        (PRIORITY.high, _("high")), (PRIORITY.now, _("now"))]
    STATUS_CHOICES = [(STATUS.sent, _("sent")), (STATUS.failed, _("failed")),
                      (STATUS.queued, _("queued")), (STATUS.requeued, _("requeued"))]

    from_email = models.CharField(_("Email From"), max_length=254,
                                  validators=[validate_email_with_name])
    to = CommaSeparatedEmailField(_("Email To"))
    cc = CommaSeparatedEmailField(_("Cc"))
    bcc = CommaSeparatedEmailField(_("Bcc"))
    subject = models.CharField(_("Subject"), max_length=989, blank=True)
    message = models.TextField(_("Message"), blank=True)
    html_message = models.TextField(_("HTML Message"), blank=True)
    """
    Emails with 'queued' status will get processed by ``send_queued`` command.
    Status field will then be set to ``failed`` or ``sent`` depending on
    whether it's successfully delivered.
    """
    status = models.PositiveSmallIntegerField(
        _("Status"),
        choices=STATUS_CHOICES, db_index=True,
        blank=True, null=True)
    priority = models.PositiveSmallIntegerField(_("Priority"),
                                                choices=PRIORITY_CHOICES,
                                                blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True, db_index=True)
    last_updated = models.DateTimeField(db_index=True, auto_now=True)
    scheduled_time = models.DateTimeField(_("Scheduled Time"),
                                          blank=True, null=True, db_index=True,
                                          help_text=_("The scheduled sending time"))
    expires_at = models.DateTimeField(_("Expires"),
                                      blank=True, null=True,
                                      help_text=_("Email won't be sent after this timestamp"))
    message_id = models.CharField("Message-ID", null=True, max_length=255, editable=False)
    number_of_retries = models.PositiveIntegerField(null=True, blank=True)
    headers = JSONField(_('Headers'), blank=True, null=True)
    template = models.ForeignKey('post_office.EmailTemplate', blank=True,
                                 null=True, verbose_name=_("Email template"),
                                 on_delete=models.CASCADE)
    context = context_field_class(_('Context'), blank=True, null=True)
    backend_alias = models.CharField(_("Backend alias"), blank=True, default='',
                                     max_length=64)

    class Meta:
        app_label = 'post_office'
        verbose_name = pgettext_lazy("Email address", "Email")
        verbose_name_plural = pgettext_lazy("Email addresses", "Emails")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_email_message = None

    def __str__(self):
        return 'Email - %s' % self.pk

    def email_message(self):
        """
        Returns Django EmailMessage object for sending.
        Can also return EmailMultiAlternative or O365 Message object.
        """
        # todo should make sure the _cached email message has a backend if we're using this to send it?
        if self._cached_email_message:
            return self._cached_email_message

        return self.prepare_email_message()

    def prepare_email_message(self, connection=None):
        """
        Returns a django ``EmailMessage`` or ``EmailMultiAlternatives`` object,
        depending on whether html_message is empty.
        Returns a O365 Message object is account is set.
        """
        from seed.models.email_settings import MICROSOFT, GOOGLE, EWS_PASS

        if get_override_recipients():
            self.to = get_override_recipients()

        if self.context is not None:
            engine = get_template_engine()
            subject = engine.from_string(self.template.subject).render(self.context)
            plaintext_message = engine.from_string(self.template.content).render(self.context)
            multipart_template = engine.from_string(self.template.html_content)
            html_message = multipart_template.render(self.context)

        else:
            subject = smart_str(self.subject)
            plaintext_message = self.message
            multipart_template = None
            html_message = self.html_message

        if isinstance(self.headers, dict) or self.expires_at or self.message_id:
            headers = dict(self.headers or {})
            if self.expires_at:
                headers.update({'Expires': self.expires_at.strftime("%a, %-d %b %H:%M:%S %z")})
            if self.message_id:
                headers.update({'Message-ID': self.message_id})
        else:
            headers = None

        # construct the object to send
        msg = None
        sender = self.organization.sender
        if sender.auth and connection:
            if sender.auth.host_service == MICROSOFT:
                # connection will be a python-o365 Account object
                # msg will be a python-o365 Message object
                msg = connection.new_message(resource=sender.email_address)
                msg.sender.address = sender.email_address
                msg.subject = subject
                msg.to.add(self.to)
                msg.cc.add(self.cc)
                msg.bcc.add(self.bcc)
                if html_message:
                    msg.body = html_message
                elif plaintext_message:
                    msg.body = plaintext_message
                """
                todo
                message_headers = []
                for name, value in headers.items():
                    if name.startswith('X-') or name.startswith('x-'):
                        message_headers.append({'name': name, 'value': value})
                msg.message_headers = message_headers
                """

                # Add an inline logo header.
                if self.template and self.template.use_logo_header:
                    logo_path = os.path.join(settings.MEDIA_ROOT, self.organization.logo.filename())
                    # todo check that the cid we have found is the logo itself?
                    match = re.search(r'src="cid:([a-zA-Z0-9]*)"', html_message)
                    if match:
                        msg.attachments.add(logo_path)
                        att = msg.attachments[0]
                        att.is_inline = True
                        att.content_id = match.group(1)
                for attachment in self.attachments.all():
                    try:
                        path = os.path.join(settings.MEDIA_ROOT, attachment.file.name)  # should be the relative path
                        msg.attachments.add([(path, attachment.name)])  # must add as a list of tuples to add a custom name
                    except Exception as e:
                        self.logs.create(status=STATUS.failed, message='Error adding attachment %s: %s' % (attachment.name, str(e)), exception_type=type(e).__name__)

            elif sender.auth.host_service == EWS_PASS:
                # connection will be an exchangelib Account object
                # msg will be an exchangelib Message (EWSMessage) object

                # in EWS, must register the header before using it as a field
                # https://web.archive.org/web/20221204210117/https://mellositmusings.com/2018/12/23/sending-an-email-with-x-headers-in-ews-via-powershell/
                if 'beam_header' not in EWSMessage.supported_fields(connection.version):
                    # not sure how to create class we can use for multiple headers... only usable with X-BEAMHelpdesk-Delivered for now
                    EWSMessage.register(attr_name="beam_header", attr_cls=BEAMHeader)
                    if 'beam_header' not in EWSMessage.supported_fields(connection.version):
                        logger.error("Could not register BEAM header in EWS.")

                msg = EWSMessage(
                    account=connection,
                    folder=connection.sent,  # saves email to this folder
                    subject=subject,
                    body=HTMLBody(html_message) if html_message else plaintext_message,
                    to_recipients=self.to,
                    cc_recipients=self.cc,
                    bcc_recipients=self.bcc,
                )
                msg.save()

                if 'X-BEAMHelpdesk-Delivered' in headers and 'beam_header' in EWSMessage.supported_fields(connection.version):
                    msg.beam_header = headers['X-BEAMHelpdesk-Delivered']
                    msg.save()

                for attachment in self.attachments.all():
                    try:
                        path = os.path.join(settings.MEDIA_ROOT, attachment.file.name)
                        binary_file_content = default_storage.open(path).read()
                        file = FileAttachment(name=attachment.name, content=binary_file_content)
                        msg.attach(file)
                    except Exception as e:
                        self.logs.create(status=STATUS.failed, message='Error adding attachment %s: %s' % (attachment.name, str(e)), exception_type=type(e).__name__)

            elif sender.auth.host_service == GOOGLE:
                msg = None
            else:
                msg = None
        if not msg:
            if html_message:
                if plaintext_message:
                    msg = EmailMultiAlternatives(
                        subject=subject, body=plaintext_message, from_email=self.from_email,
                        to=self.to, bcc=self.bcc, cc=self.cc,
                        headers=headers)
                    msg.attach_alternative(html_message, "text/html")
                else:
                    msg = EmailMultiAlternatives(
                        subject=subject, body=html_message, from_email=self.from_email,
                        to=self.to, bcc=self.bcc, cc=self.cc,
                        headers=headers)
                    msg.content_subtype = 'html'
                if hasattr(multipart_template, 'attach_related'):
                    multipart_template.attach_related(msg)
            else:
                msg = EmailMessage(
                    subject=subject, body=plaintext_message, from_email=self.from_email,
                    to=self.to, bcc=self.bcc, cc=self.cc,
                    headers=headers)
            if connection:
                # connection should be already supplied by setup_basic_backend
                msg.connection = connection

            for attachment in self.attachments.all():
                try:
                    if attachment.headers:
                        mime_part = MIMENonMultipart(*attachment.mimetype.split('/'))
                        mime_part.set_payload(attachment.file.read())
                        for key, val in attachment.headers.items():
                            try:
                                mime_part.replace_header(key, val)
                            except KeyError:
                                mime_part.add_header(key, val)
                        msg.attach(mime_part)
                    else:
                        msg.attach(attachment.name, attachment.file.read(), mimetype=attachment.mimetype or None)
                    attachment.file.close()
                except Exception as e:
                    self.logs.create(status=STATUS.failed, message='Error adding attachment %s: %s' % (attachment.name, str(e)), exception_type=type(e).__name__)

        self._cached_email_message = msg
        return msg

    def dispatch(self, log_level=None,
                 disconnect_after_delivery=True, commit=True):
        """
        Sends email and log the result.
        """
        try:
            mail = self.email_message()
        except Exception:
            if commit:
                status = STATUS.failed
                message = 'Failed to retrieve email message.'
                exception_type = ''
                logger.exception('Failed to retrieve email message.')
            else:
                # If run in a bulk sending mode, re-raise and let the outer layer handle the exception
                raise
        else:
            try:
                result = mail.send()
            except Exception as e:
                if commit:
                    status = STATUS.failed
                    message = str(e)
                    exception_type = type(e).__name__
                    logger.exception('Failed to send email')
                else:
                    raise
            else:
                if result == 1 or result is None:  # regular mail and Microsoft return 1; EWS returns None
                    status = STATUS.sent
                    message = ''
                    exception_type = ''
                else:
                    if commit:
                        status = STATUS.failed
                        message = 'Sending failed without an exception.'
                        exception_type = ''
                        logger.exception('Failed to send email without exception')
                    else:
                        raise RuntimeError('Failed to send email without exception')

        if disconnect_after_delivery:
            connections.close()

        if commit:
            self.status = status
            self.save(update_fields=['status'])

            if log_level is None:
                log_level = get_log_level()

            # If log level is 0, log nothing, 1 logs only sending failures
            # and 2 means log both successes and failures
            if log_level == 1 and status == STATUS.failed:
                self.logs.create(status=status, message=message, exception_type=exception_type)
            elif log_level == 2:
                self.logs.create(status=status, message=message, exception_type=exception_type)

        return status

    def clean(self):
        if self.scheduled_time and self.expires_at and self.scheduled_time > self.expires_at:
            raise ValidationError(_("The scheduled time may not be later than the expires time."))

    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)


class Log(models.Model):
    """
    A model to record sending email sending activities.
    """

    STATUS_CHOICES = [(STATUS.sent, _("sent")), (STATUS.failed, _("failed"))]

    email = models.ForeignKey(Email, editable=False, related_name='logs',
                              verbose_name=_('Email address'), on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    status = models.PositiveSmallIntegerField(_('Status'), choices=STATUS_CHOICES)
    exception_type = models.CharField(_('Exception type'), max_length=255, blank=True)
    message = models.TextField(_('Message'))

    class Meta:
        app_label = 'post_office'
        verbose_name = _("Log")
        verbose_name_plural = _("Logs")

    def __str__(self):
        return str(self.date)


class EmailTemplateManager(models.Manager):
    def get_by_natural_key(self, name, language, default_template):
        return self.get(name=name, language=language, default_template=default_template)


class EmailTemplate(models.Model):
    """
    Model to hold template information from db
    """
    # BEAM
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, blank=True, null=True, related_name='email_templates')
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True, related_name='email_templates')
    use_logo_header = models.BooleanField(_('Use Org Logo in Header'), blank=True, default=False,
                                          help_text=_('If this is checked, it will add the Logo into the Email Banner'))
    use_name_footer = models.BooleanField(_('Use Org Name in Footer'), blank=True, default=False,
                                          help_text=_(
                                              'If this is checked, it will add the Org name into the Email Footer'))
    use_color_header = models.BooleanField(_('Use Org Color in Header'), blank=True, default=False,
                                           help_text=_(
                                               'If this is checked, it will add the Orgs Color into the Email Banner'))
    use_color_footer = models.BooleanField(_('Use Org Color in Footer'), blank=True, default=False,
                                           help_text=_(
                                               'If this is checked, it will add the Orgs Color into the Email Footer'))
    tag_mappings = models.JSONField(default=dict, blank=True)
    # end BEAM

    name = models.CharField(_('Name'), max_length=255, help_text=_("e.g: 'welcome_email'"))
    description = models.TextField(_('Description'), blank=True,
                                   help_text=_("Description of this template."))
    created = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    subject = models.CharField(max_length=255, blank=True,
        verbose_name=_("Subject"), validators=[validate_template_syntax])
    content = models.TextField(blank=True,
        verbose_name=_("Content"), validators=[validate_template_syntax])
    html_content = models.TextField(blank=True,
        verbose_name=_("HTML content"), validators=[validate_template_syntax])
    language = models.CharField(max_length=12,
        verbose_name=_("Language"),
        help_text=_("Render template in alternative language"),
        default='', blank=True)
    default_template = models.ForeignKey('self', related_name='translated_templates',
        null=True, default=None, verbose_name=_('Default template'), on_delete=models.CASCADE)

    objects = EmailTemplateManager()

    class Meta:
        app_label = 'post_office'
        unique_together = ('name', 'language', 'default_template')
        verbose_name = _("Email Template")
        verbose_name_plural = _("Email Templates")
        ordering = ['name']

    def __str__(self):
        return '%s %s' % (self.name, self.language)

    def natural_key(self):
        return (self.name, self.language, self.default_template)

    def save(self, *args, **kwargs):
        # If template is a translation, use default template's name
        if self.default_template and not self.name:
            self.name = self.default_template.name

        template = super().save(*args, **kwargs)
        cache.delete(self.name)
        return template

    def get_context_names(self):
        """
        Go through text/html content and find instances where the template uses {{ tag }}. In some cases,
        tag will be var[i] as tag didn't follow the correct formatting. Replace var[i] for stored tag in tag_mappings
        :return: list, all instances of tags
        """
        tag_mappings = {v: k for k, v in self.tag_mappings.items()}
        matches = []
        for match in re.findall(r'{{[ ]*[^} \n]*[ ]*}}', self.subject + self.content):
            match = match.strip(' {}')
            if match in tag_mappings:
                match = tag_mappings[match]
            if match not in matches:
                matches.append(match)

        return matches

    def build_tag_var_context(self, context):
        """
        When about to send an email, build an updated context using var[i] names since Django would probably still error
        if we revert back to original tag names before sending
        :param context: Dict, populated context being used to send emails
        :return: dict, an updated context containing mapping of var[i] to context[original tag name]
        """
        new_context = {self.tag_mappings[tag]: context[tag] for tag in self.tag_mappings if tag in context}
        return new_context

    @classmethod
    def get_tag_var_dict(cls, data):
        """
        Parse subject, and html_content to create a dict of {tag: var[i]} if the tag is not valid
        :param data: The data from request.data made during an api call
        :return: dict: A dict of mappings between tag name to var[i]
        """
        subject = data.get('subject', '')
        content = data.get('html_content', '')
        counter = 0
        tag_mappings = {}

        for match in re.findall('{{[ ]*[^}\n]*[ ]*}}', subject + content):
            match = match.strip(' {}')  # Remove starting and ending tags/spaces
            if not check_if_context_appropriate(match) and match not in tag_mappings:
                tag_mappings[match] = 'var' + str(counter)
                counter += 1

        return tag_mappings

    @classmethod
    def replace_tags(cls, data, tag_mappings, reverse=False):
        """
        Parse subject, html_content, and content to replace tags in tag_mappings with their var[i] form.
        tag_mappings can be form {tag: var[i]}, or if reverse is true, it has the reverse pairing form ie var[i] maps to tag
        :param data: The data from request.data made during an api call
        :param tag_mappings: The mapping of tag to var[i]
        :param reverse: whether to reverse the tag_mappings mapping to replace var[i]s for tags
        :return: dict: A dict of mappings between tag name to var[i]
        """
        if reverse:
            tag_mappings = {v: k for k, v in tag_mappings.items()}

        for attr in ['subject', 'html_content', 'content']:
            if attr in data:
                for k, v in tag_mappings.items():
                    data[attr] = re.sub(r'{{[ ]*' + re.escape(k) + r'[ ]*}}', '{{ ' + v + ' }}', data[attr])

        return data

    def __str__(self):
        return 'EmailTemplate - %s' % self.pk


def get_upload_path(instance, filename):
    """Overriding to store the original filename"""
    if not instance.name:
        instance.name = filename  # set original filename
    date = timezone.now().date()
    filename = '{name}.{ext}'.format(name=uuid4().hex,
                                     ext=filename.split('.')[-1])

    return os.path.join('post_office_attachments', str(date.year),
                        str(date.month), str(date.day), filename)


class Attachment(models.Model):
    """
    A model describing an email attachment.
    """
    file = models.FileField(_('File'), upload_to=get_upload_path)
    name = models.CharField(_('Name'), max_length=255, help_text=_("The original filename"))
    emails = models.ManyToManyField(Email, related_name='attachments',
                                    verbose_name=_('Emails'))
    mimetype = models.CharField(max_length=255, default='', blank=True)
    headers = JSONField(_('Headers'), blank=True, null=True)

    class Meta:
        app_label = 'post_office'
        verbose_name = _("Attachment")
        verbose_name_plural = _("Attachments")

    def __str__(self):
        return self.name
