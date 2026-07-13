"""Golden request/response JSON pairs for every worked example in
``docs/WEBSERVICE_API.md`` (Milestone 6, ``docs/WEBSERVICE_BUILD.md`` §6).

Each ``pairs/*.json`` file is ``{method, path, status, request, response}``, generated
from a real (deterministic, seeded) call through the actual service -- exactly the
values it returns today -- so these lock the wire format the way the library's own
tests lock known designs and injected effects (``CLAUDE.md``). ``test_contract.py``
POSTs every pair's ``request`` and compares the live response to the stored
``response`` structurally: exact for strings/bools/nulls, ``pytest.approx`` for
numbers, recursive for objects/arrays (``support.assert_matches``).
"""
