from gpiozero import Buzzer, LED, Servo


class GPIO:
    def __init__(self):
        self.buzzer = Buzzer(21)
        self.red_led = LED(14)
        self.green_led = LED(15)
        self.servo = Servo(17)

    def green(self):
        self.servo.max()
        self.buzzer.off()
        self.red_led.off()
        self.green_led.on()

    def red(self):
        self.servo.mid()
        self.buzzer.on()
        self.red_led.on()
        self.green_led.off()
