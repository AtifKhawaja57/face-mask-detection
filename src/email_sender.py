import smtplib
import ssl
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import datetime
import os


class EmailSender:
    def __init__(self):
        self.sender_email = os.environ.get('EMAIL_ID')
        self.receiver_email = os.environ.get('EMAIL_ID')
        self.password = os.environ.get('EMAIL_PWD')

    def send_message(self, body_temperature):
        message = MIMEMultipart("alternative")
        message["Subject"] = "Alert: A New Person Entered the Premises"
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        with open(os.path.join(os.path.dirname(__file__), "demo.jpg"), 'rb') as f:
            mime = MIMEBase('image', 'jpg', filename=os.path.join(os.path.dirname(__file__) + "demo.jpg"))
            mime.add_header('Content-Disposition', 'attachment', filename=os.path.join(os.path.dirname(__file__) + "demo.jpg"))
            mime.add_header('X-Attachment-Id', '0')
            mime.add_header('Content-ID', '<0>')
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            message.attach(mime)

        content = f'''
        <html>
            <body>
                <h1>Alert</h1>
                <h2>A new has Person entered the Premises</h2>
                <h2>Body Temperature: {body_temperature}</h2>
                <h2>Mask On?: No/h2>
                <h2>Time: {datetime.datetime.now().strftime("%H:%M:%S")}</h2>
                <p>
                    <img src="cid:0">
                </p>
            </body>
        </html>'''
        body = MIMEText(content, 'html')
        message.attach(body)
        # Create secure connection with server and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(
                self.sender_email, self.receiver_email, message.as_string()
            )