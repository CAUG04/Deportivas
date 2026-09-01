"""Root conftest.py.

Its only job is to exist here: pytest always inserts a discovered
conftest.py's directory onto ``sys.path``, which is what lets test modules
import shared fixtures as ``tests.fixtures.<module>`` (namespace packages,
no ``__init__.py`` needed) instead of duplicating synthetic schemas per
test file.
"""
