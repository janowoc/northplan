"""HTTP layer. Thin by construction.

Translates JSON to engine calls and engine results back to JSON, and serves
``web/`` as static files. No financial logic lives here.

This is also where real dollars become nominal, if the client asks for nominal:
the engine never returns a nominal figure, so the conversion happens here,
once, on the way out.
"""
