# Dashboard Frontend

The ACN dashboard is implemented in `apps/web` as a typed React application.

## Views

- Commit graph
- Branch graph
- Metrics timeline
- Experiment inspector
- Controller decisions
- Rollback history
- Live logs
- Override console

## Integration Contract

REST:

- `GET /api/v1/dashboard/snapshot`
- `POST /api/v1/overrides`

Live updates:

- `GET /api/v1/dashboard/events` for SSE
- `WS /api/v1/dashboard/ws` as fallback

The frontend does not depend on fake data. If backend data is not available, views render empty or integration-error states.

## Snapshot Shape

The snapshot contains:

- commit graph nodes and edges
- branch graph nodes and edges
- metrics timeline
- experiment summaries
- controller decisions
- rollback history
- live log entries

