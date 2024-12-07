from typing import Dict, Optional
from .security import UserInDB, get_password_hash

class UserStore:
    def __init__(self):
        self._users: Dict[str, UserInDB] = {
            # Default admin user
            "admin": UserInDB(
                username="admin",
                email="admin@example.com",
                full_name="Admin User",
                disabled=False,
                role="admin",
                hashed_password=get_password_hash("admin123")
            )
        }

    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self._users.get(username)

    def create_user(self, username: str, password: str, email: str, full_name: str) -> UserInDB:
        """Create a new user."""
        if username in self._users:
            raise ValueError("Username already exists")
        
        user = UserInDB(
            username=username,
            email=email,
            full_name=full_name,
            disabled=False,
            role="user",  # Default role for new users
            hashed_password=get_password_hash(password)
        )
        self._users[username] = user
        return user

    def list_users(self) -> list[UserInDB]:
        """List all users."""
        return list(self._users.values())

    def update_user(self, username: str, **kwargs) -> Optional[UserInDB]:
        """Update user details."""
        if username not in self._users:
            return None
        
        user = self._users[username]
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        return user

    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username not in self._users or username == "admin":
            return False
        del self._users[username]
        return True

# Global user store instance
user_store = UserStore()
