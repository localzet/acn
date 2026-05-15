from acn.config.settings import Settings


def test_settings_builds_database_url() -> None:
    settings = Settings(
        postgres_host="db",
        postgres_port=5433,
        postgres_db="acn_test",
        postgres_user="user",
        postgres_password="password",  # noqa: S106
    )

    assert settings.database_url == "postgresql+psycopg://user:password@db:5433/acn_test"


def test_settings_builds_redis_url() -> None:
    settings = Settings(redis_host="cache", redis_port=6380, redis_db=2)

    assert str(settings.redis_url) == "redis://cache:6380/2"
