# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Context Switcher MCP seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them by email to [code-developer@democratize.technology](mailto:code-developer@democratize.technology).

Please include the following information in your report (as much as you can provide):

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Response Timeline

We will acknowledge receipt of your vulnerability report within 48 hours and will send a more detailed response within 96 hours indicating the next steps in handling your report.

After the initial reply to your report, we will endeavor to keep you informed of the progress being made towards a fix and full announcement, and may ask for additional information or guidance surrounding the reported issue.

## Security Considerations

Context Switcher MCP is a Model Context Protocol (MCP) server that:

* **Executes LLM queries**: Makes network requests to AI services (AWS Bedrock, LiteLLM, Ollama)
* **Processes user input**: Handles structured queries and commands from MCP clients
* **Manages sessions**: Stores temporary session state and perspective data
* **File operations**: Limited file system access for circuit breaker state persistence

### Areas of Concern

When reporting vulnerabilities, pay special attention to:

1. **Injection vulnerabilities** in prompt construction or query parameters
2. **Path traversal** attacks in file operations
3. **Deserialization** vulnerabilities in session management
4. **Resource exhaustion** through malformed or excessive requests
5. **Information disclosure** through error messages or logs
6. **Authentication bypass** in session management
7. **Race conditions** in concurrent operations

### Security Measures

Current security measures include:

* Input validation on all user-provided parameters
* Path validation to prevent directory traversal
* Circuit breaker pattern for external service resilience
* Secure session token generation and validation
* Thread-safe concurrent operation design
* Comprehensive error handling without information leakage

## Responsible Disclosure

We ask that you:

* Give us reasonable time to fix the issue before public disclosure
* Make a good faith effort to avoid privacy violations, destruction of data, and interruption or degradation of our services
* Only interact with accounts you own or with explicit permission of the account holder

## Recognition

We value the security community and will acknowledge security researchers who report vulnerabilities to us in a responsible manner. With your permission, we will publicly acknowledge your contribution in our release notes when we deploy a fix.

## Contact

For any questions about this security policy, please contact [code-developer@democratize.technology](mailto:code-developer@democratize.technology).

---

*This security policy is based on industry best practices and is regularly reviewed and updated.*
