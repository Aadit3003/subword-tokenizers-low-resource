import time


class Timer:

    def __init__(self):
        self._start_time = None
        self._stop_time = None
        self._is_running = False


    def __str__(self):
        """Convert a timer result to a string."""

        if self._start_time is None:
            raise Exception(
                'The timer cannot be read because it has not been run.')

        if self._stop_time is None:
            raise Exception(
                  'The timer cannot be read while it is running.')

        return('{:.3f} secs'.format(self._stop_time - self._start_time))


    def start(self):
        """Start the timer."""

        if self._is_running:
            raise Exception(
                'Timer is already running. Use .stop() to stop it.')

        self._start_time = time.perf_counter()
        self._stop_time = None
        self._is_running = True


    def stop(self):
        """Stop the timer."""

        if not self._is_running:
            raise Exception(
                'Timer is not running. Use .start() to start it.')

        self._stop_time = time.perf_counter()
        self._is_running = False
