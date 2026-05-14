from enum import Enum
from typing import List, Dict

class UserRole(str, Enum):
    FARMER = "farmer"
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"

# Mapping Role -> Allowed Domains (A1, A2, A3, A4)
# A1: Farm Ops
# A2: Warehouse/Inventory
# A3: Analytics/Report
# A4: IoT/Devices
ROLE_PERMISSIONS: Dict[UserRole, List[str]] = {
    UserRole.FARMER: ["a1", "a4"],
    UserRole.EMPLOYEE: ["a1", "a2","a3", "a4"],
    UserRole.MANAGER: ["a1", "a2", "a3", "a4"],
    UserRole.ADMIN: ["a1", "a2", "a3", "a4"],
}

def can_access_domain(role: str, domain: str) -> bool:
    """Check if a role has permission to access a specific domain."""
    try:
        user_role = UserRole(role.lower())
        allowed_domains = ROLE_PERMISSIONS.get(user_role, [])
        return domain.lower() in allowed_domains
    except ValueError:
        # If role is invalid, default to no access
        return False
