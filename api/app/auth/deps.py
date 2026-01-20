from fastapi import Request, HTTPException

def require_user(request: Request) -> dict:
    """
    Session-based auth dependency.
    Session payload should store:
      {"username": "...", "role": "admin|reviewer|viewer"}
    """
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

def require_role(*roles: str):
    def _dep(request: Request) -> dict:
        user = require_user(request)
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return _dep
