#!/usr/bin/env python3
"""Create a MENET Agent account from the deployment environment."""

import argparse
import getpass
import os
from pathlib import Path

from agent.store import TaskStore


parser = argparse.ArgumentParser(description="Create a MENET Agent user account")
parser.add_argument("username")
parser.add_argument("display_name")
parser.add_argument("--role", choices=("user", "admin"), default="user")
args = parser.parse_args()
password = getpass.getpass("Password (min 8 characters): ")
database = Path(os.environ.get("MENET_AGENT_DB", "runs/agent.db"))
user = TaskStore(str(database)).create_account(args.username, password, args.display_name, args.role)
print(f"created {user['username']} ({user['user_id']}, role={user['role']})")
