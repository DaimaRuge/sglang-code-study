# Security, License And Supply Chain

*Full content available in the original report file.*

## License: Apache 2.0 - Commercial Friendly

## Security Assessment: C+ (Needs hardening for production)

- No built-in authentication (opt-in via --api-key)
- No rate limiting
- trust_remote_code default True

## Production Security Checklist (Top 5)

1. Set --api-key + TLS reverse proxy
2. Restrict network binding
3. Disable trust_remote_code in production
4. Set log level >= warning
5. Run vulnerability scan
