# Migrations

Alembic migration scripts live here.

Setup (once the DB is configured):

```bash
pip install alembic
alembic init packages/database/migrations
# point alembic.ini's sqlalchemy.url at DATABASE_URL and
# set target_metadata = Base.metadata (from packages/database/models.py)
alembic revision --autogenerate -m "initial tables"
alembic upgrade head
```
