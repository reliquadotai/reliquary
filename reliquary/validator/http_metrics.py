"""Bounded route metrics and independent admission control for diagnostic reads."""
import time

from starlette.responses import JSONResponse


class HttpMetrics:
    BUCKETS = (0.005, 0.025, 0.1, 0.25, 1.0, 5.0)

    def __init__(self):
        self.routes = {}
        self.diagnostic_active = 0

    @staticmethod
    def route(path):
        if path.startswith('/miner-verdict-history/'):
            return 'verdict_history'
        if path.startswith('/miner-verdicts/'):
            return 'verdict_lookup' if len(path.strip('/').split('/')) == 4 else 'verdict_feed'
        if path.startswith('/verdicts/'):
            return 'legacy_verdicts'
        return path if path in {'/state', '/miner-state', '/health', '/livez', '/readyz',
                                '/submit', '/submit/precommit', '/http-metrics', '/checkpoint',
                                '/runtime-contract'} else 'other'

    def snapshot(self):
        return {'observed_at': time.time(), 'duration_bucket_bounds_seconds': self.BUCKETS,
                'diagnostic_active': self.diagnostic_active,
                'routes': {k: {**v, 'duration_buckets': list(v['duration_buckets']),
                              'status_classes': dict(v['status_classes'])} for k, v in self.routes.items()}}


class HttpMetricsMiddleware:
    def __init__(self, app, metrics, diagnostic_limit=16):
        self.app, self.metrics, self.limit = app, metrics, diagnostic_limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        route = self.metrics.route(scope.get('path', ''))
        method = scope.get('method', 'OTHER')
        key = f"{method if method in ('GET', 'POST') else 'OTHER'} {route}"
        row = self.metrics.routes.setdefault(key, {'requests': 0, 'active': 0, 'peak_active': 0,
            'response_bytes': 0, 'duration_seconds': 0.0, 'duration_buckets': [0] * 7,
            'status_classes': {}, 'overload_rejections': 0})
        row['requests'] += 1
        row['active'] += 1
        row['peak_active'] = max(row['peak_active'], row['active'])
        started, status, size, admitted = time.perf_counter(), 500, 0, False

        async def capture(message):
            nonlocal status, size
            if message['type'] == 'http.response.start':
                status = message['status']
            elif message['type'] == 'http.response.body':
                size += len(message.get('body', b''))
            await send(message)

        try:
            if method == 'GET' and route in {'verdict_history', 'verdict_lookup', 'verdict_feed', 'legacy_verdicts'}:
                if self.metrics.diagnostic_active >= self.limit:
                    row['overload_rejections'] += 1
                    return await JSONResponse({'detail': 'diagnostic_capacity'}, status_code=503,
                        headers={'Retry-After': '1', 'Cache-Control': 'no-store'})(scope, receive, capture)
                self.metrics.diagnostic_active += 1
                admitted = True
            await self.app(scope, receive, capture)
        finally:
            if admitted:
                self.metrics.diagnostic_active -= 1
            elapsed = time.perf_counter() - started
            row['active'] -= 1
            row['response_bytes'] += size
            row['duration_seconds'] += elapsed
            bucket = next((i for i, upper in enumerate(self.metrics.BUCKETS) if elapsed <= upper), 6)
            row['duration_buckets'][bucket] += 1
            group = str(status // 100) + 'xx'
            row['status_classes'][group] = row['status_classes'].get(group, 0) + 1
