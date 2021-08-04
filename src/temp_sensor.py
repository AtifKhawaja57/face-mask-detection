from smbus2 import SMBus
from mlx90614 import MLX90614


class TempSensor:
    def __init__(self):
        self.bus = SMBus(1)
        self.sensor = MLX90614(self.bus, address=0x5A)

    def temperature(self):
        return self.sensor.get_object_1()

    def cleanup(self):
        self.bus.close()


