from .time import Time


class Model:

    def __init__(self):
        self.time = Time()
        self.trigger_calls = []
        self.model_init_time = self.time.get()

    def trigger(self):
        self.trigger_calls.append(self.time.get())

    def prev_trigger_call_time(self):
        """
        Получить время прошлого запуска функции trigger(),
        либо время создания модели, если запусков trigger() еще не было
        :return: время прошлого запуска
        """
        if len(self.trigger_calls) == 1:
            return self.model_init_time
        return self.trigger_calls[-2]
