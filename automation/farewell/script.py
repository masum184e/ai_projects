import smtplib
from email.message import EmailMessage

# Your Gmail credentials
EMAIL_ADDRESS = "masum1834e@gmail.com"
EMAIL_PASSWORD = "abcdefghijklmnopqrstuvwxyz"  # Use Gmail App Password

# Load HTML template (contains {greeting_name})
with open("./email.html", "r", encoding="utf-8") as f:
    html_template = f.read()

# Recipient list with name, email, and gender
receivers = [
    {'id': 1, 'name': 'Jonathan Gregory', 'email': 'jonathangregory4671@gmail.com', 'gender': 'male'},
    {'id': 2, 'name': 'Kaylee Cook', 'email': 'kayleecook2723@gmail.com', 'gender': 'female'},
    {'id': 3, 'name': 'Ryan Barry', 'email': 'ryanbarry1526@gmail.com', 'gender': 'male'},
    {'id': 4, 'name': 'Dylan Nelson', 'email': 'dylannelson1065@gmail.com', 'gender': 'male'},
    {'id': 5, 'name': 'Jamie Terry', 'email': 'jamieterry8896@gmail.com', 'gender': 'female'},
    {'id': 6, 'name': 'Mark Rice', 'email': 'markrice1085@gmail.com', 'gender': 'male'},
    {'id': 7, 'name': 'Sophia Baker', 'email': 'sophiabaker1353@gmail.com', 'gender': 'female'},
    {'id': 8, 'name': 'Kyle Bennett', 'email': 'kylebennett6261@gmail.com', 'gender': 'male'},
    {'id': 9, 'name': 'Taylor Watson', 'email': 'taylorwatson2424@gmail.com', 'gender': 'female'},
    {'id': 10, 'name': 'Emma Morrison', 'email': 'emmamorrison9876@gmail.com', 'gender': 'female'}
]


# Send via Gmail SMTP
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

    for person in receivers:
        # Create greeting name
        suffix = "Vai" if person["gender"] == "male" else "Apu"
        greeting_name = f"{person['name']} {suffix}"

        # Replace {greeting_name} in the HTML template
        personalized_html = html_template.replace("{greeting_name}", greeting_name)

        # Create and send email
        msg = EmailMessage()
        msg['Subject'] = "Invitation to “অনুবর্তন ৩.০” – Farewell Ceremony"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = person['email']
        msg.set_content("This is an HTML email. Please view it in an HTML-compatible client.")
        msg.add_alternative(personalized_html, subtype='html')

        smtp.send_message(msg)
        print(f"✅ Email sent to {person["id"]} {greeting_name} successfully!")
