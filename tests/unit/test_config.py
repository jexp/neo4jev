import pytest

from neo4jev.config import Settings


REQUIRED_ENV = {
    "NEO4J_USERNAME": "companies",
    "NEO4J_PASSWORD": "companies",
    "NEO4J_DATABASE": "companies",
    "TYPESAFE_API_KEY": "test-key",
}


def _set_env(monkeypatch, **overrides):
    for key, value in {**REQUIRED_ENV, **overrides}.items():
        monkeypatch.setenv(key, value)


def test_settings_from_neo4j_uri(monkeypatch):
    _set_env(monkeypatch, NEO4J_URI="neo4j+s://demo.neo4jlabs.com:7687")

    settings = Settings.from_env(env_file="/nonexistent")

    assert settings.neo4j_uri == "neo4j+s://demo.neo4jlabs.com:7687"
    assert settings.neo4j_username == "companies"
    assert settings.neo4j_password == "companies"
    assert settings.neo4j_database == "companies"
    assert settings.typesafe_api_key == "test-key"


def test_settings_from_neo4j_url(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    _set_env(monkeypatch, NEO4J_URL="neo4j+s://demo.neo4jlabs.com:7687")

    settings = Settings.from_env(env_file="/nonexistent")

    assert settings.neo4j_uri == "neo4j+s://demo.neo4jlabs.com:7687"


def test_neo4j_uri_takes_precedence_over_neo4j_url(monkeypatch):
    _set_env(
        monkeypatch,
        NEO4J_URI="neo4j+s://uri-wins:7687",
        NEO4J_URL="neo4j+s://url-loses:7687",
    )

    settings = Settings.from_env(env_file="/nonexistent")

    assert settings.neo4j_uri == "neo4j+s://uri-wins:7687"


def test_missing_neo4j_uri_and_url_raises(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_URL", raising=False)
    _set_env(monkeypatch)

    with pytest.raises(ValueError, match="NEO4J_URI"):
        Settings.from_env(env_file="/nonexistent")


def test_missing_typesafe_api_key_raises(monkeypatch):
    _set_env(monkeypatch, NEO4J_URI="neo4j+s://demo.neo4jlabs.com:7687")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        Settings.from_env(env_file="/nonexistent")


def test_missing_typesafe_api_key_tolerated_when_not_required(monkeypatch):
    _set_env(monkeypatch, NEO4J_URI="neo4j+s://demo.neo4jlabs.com:7687")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    settings = Settings.from_env(env_file="/nonexistent", require_typesafe_key=False)

    assert settings.typesafe_api_key == ""
    assert settings.neo4j_uri == "neo4j+s://demo.neo4jlabs.com:7687"


def test_from_env_without_key_requirement_accepts_url_alias_and_requires_uri(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    _set_env(monkeypatch, NEO4J_URL="neo4j+s://demo.neo4jlabs.com:7687")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    settings = Settings.from_env(env_file="/nonexistent", require_typesafe_key=False)

    assert settings.neo4j_uri == "neo4j+s://demo.neo4jlabs.com:7687"
    assert settings.typesafe_api_key == ""

    monkeypatch.delenv("NEO4J_URL", raising=False)
    with pytest.raises(ValueError, match="NEO4J_URI"):
        Settings.from_env(env_file="/nonexistent", require_typesafe_key=False)


def test_from_env_without_key_requirement_keeps_a_present_key(monkeypatch):
    _set_env(monkeypatch, NEO4J_URI="neo4j+s://demo.neo4jlabs.com:7687")

    settings = Settings.from_env(env_file="/nonexistent", require_typesafe_key=False)

    assert settings.typesafe_api_key == "test-key"
