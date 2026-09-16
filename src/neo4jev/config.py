from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    typesafe_api_key: str

    @classmethod
    def from_env(
        cls, *, env_file: str | None = None, require_typesafe_key: bool = True
    ) -> "Settings":
        load_dotenv(dotenv_path=env_file)

        uri = os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_URL")
        if not uri:
            raise ValueError("Missing required environment variable: NEO4J_URI (or NEO4J_URL)")

        return cls(
            neo4j_uri=uri,
            neo4j_username=_require_env("NEO4J_USERNAME"),
            neo4j_password=_require_env("NEO4J_PASSWORD"),
            neo4j_database=_require_env("NEO4J_DATABASE"),
            typesafe_api_key=(
                _require_env("TYPESAFE_API_KEY")
                if require_typesafe_key
                else os.environ.get("TYPESAFE_API_KEY", "")
            ),
        )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value
