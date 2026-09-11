# Security Trimming for a SharePoint-Based Enterprise Knowledge Graph

## Executive Summary

To preserve SharePoint security within an Enterprise Knowledge Graph, the recommended approach is to implement **security trimming**.

Instead of materializing every authorization relationship as graph edges, documents are tagged with the users and groups authorized to access them. During retrieval, the system determines the user's current identities and filters candidate documents accordingly.

This is the approach adopted by most enterprise search platforms because it scales efficiently, minimizes graph complexity, and remains aligned with the source system's security model.

---

# Why Security Trimming?

## The Scalability Problem

A naïve authorization model stores permissions as graph relationships:

```text
(User)-[:CAN_READ]->(Document)
```

For a large repository:

- Millions of documents
- Tens of thousands of users
- Frequent group membership changes

the number of authorization edges can become extremely large.

In practice, most enterprise search solutions avoid this explosion by storing permissions as document metadata rather than graph edges.

---

## Industry Practice

Security trimming is the dominant approach used by:

- Microsoft Search
- Azure AI Search
- SharePoint Search
- Elasticsearch Document Level Security (DLS)
- OpenSearch Security
- Many enterprise RAG platforms

The principle is simple:

```text
Document
    Allowed Principals
         |
         V

User
    Current Principals
```

Access is granted when:

```text
Document.AllowedPrincipals
    ∩
User.Principals
≠ ∅
```

where "principal" means:

- User
- Entra ID Group
- SharePoint Group
- Other security identity

---

# Proposed Model for Kortex

## Graph Entities

```text
(User)

(Group)

(Document)

(Chunk)

(Entity)
```

Relationships:

```text
(User)-[:MEMBER_OF]->(Group)

(Document)-[:HAS_CHUNK]->(Chunk)

(Chunk)-[:MENTIONS]->(Entity)
```

## Document Security Metadata

Each document stores:

```json
{
  "id": "DOC-123",
  "url": "...",
  "allowed_principals": [
      "group:03d8e370",
      "group:9e2f7b44",
      "user:5d8a7ca1"
  ]
}
```

Important:

- Store Entra Object IDs, not names.
- Do not copy ACLs onto chunks.
- Chunks inherit permissions from their parent document.

This keeps the graph compact.

---

# Implementation in Python

## Step 1: Extract SharePoint Permissions During Indexing

When ingesting a document:

```http
GET /sites/{site-id}/drive/items/{item-id}/permissions
```

Using the Microsoft Graph SDK:

```python
permissions = await client.sites.by_site_id(site_id).drive.items.by_drive_item_id(item_id).permissions.get()
```

The indexer extracts:

```text
- Entra Groups
- Users
- Site Members
- Site Owners
- Site Visitors
```

and builds:

```python
allowed_principals = ["group:03d8e370", "group:9e2f7b44", "user:5d8a7ca1"]
```

which is stored on the Document node.

---

## Step 2: Resolve User Memberships

At authentication time, determine the user's groups:

```http
POST /users/{user-id}/getMemberGroups
```

Python:

```python
groups = await client.users.by_user_id(user_id).get_member_groups.post(body={"securityEnabledOnly": False})
```

Build:

```python
user_principals = {"user:a1b2c3", "group:03d8e370", "group:9e2f7b44", "group:5e1c7dd1"}
```

---

## Step 3: Cache Memberships

Group resolution should not occur for every document.

Typical implementation:

```python
user_cache[user_id] = {"principals": user_principals, "expiry": now + timedelta(hours=1)}
```

The cache is refreshed periodically.

---

## Step 4: Security Filter Retrieved Documents

After graph retrieval:

```python
def can_access(document, user_principals):
    return bool(set(document.allowed_principals) & user_principals)
```

Usage:

```python
authorized_docs = [doc for doc in candidate_docs if can_access(doc, user_principals)]
```

Only authorized documents are provided to the LLM.

---

# Authorization Flow

```text
User Question
      |
      V

Graph Retrieval
      |
      V

Candidate Chunks
      |
      V

Parent Documents
      |
      V

Security Filter
      |
      V

Authorized Documents
      |
      V

LLM Answer
```

The LLM never sees information from unauthorized documents.

---

# Comparison with Option 1

## Option 1: Authorization Graph

Model:

```text
(User)-[:MEMBER_OF]->(Group)

(Group)-[:CAN_READ]->(Document)
```

### Advantages

- Pure graph representation.
- Cypher-native authorization queries.
- Easy visualization.
- Useful for governance analysis.
- Useful for impact analysis.

### Drawbacks

- Large number of relationships.
- Synchronization overhead.
- Potentially expensive traversals.
- More complex ingestion pipeline.

---

## Option 4: Security Trimming

Model:

```text
Document.allowed_principals
```

### Advantages

- Industry-standard approach.
- Very scalable.
- Small graph footprint.
- Fast authorization checks.
- Compatible with millions of chunks.
- Easy to cache.

### Drawbacks

- Some authorization logic occurs outside Cypher.
- Less convenient for governance analytics.

---

# Hybrid Architecture

The most practical design is a combination of both approaches.

Store:

```text
(User)-[:MEMBER_OF]->(Group)
```

inside the graph because:

- Organizational reasoning becomes possible.
- Governance queries remain possible.
- Agents can navigate organizational structures.

Store:

```text
Document.allowed_principals
```

as document metadata because:

- Retrieval remains fast.
- Security trimming is efficient.
- No explosion of authorization edges.

Result:

```text
(User)-[:MEMBER_OF]->(Group)

Document.allowed_principals
```

Authorization becomes:

```python
document.allowed_principals
    ∩
user_principals
```

while the graph continues to provide rich semantics for agentic navigation and reasoning.

---

# Recommendation

For Kortex, the recommended architecture is:

1. Store SharePoint ACLs as `allowed_principals` on Document nodes.
2. Resolve user memberships through Microsoft Graph `getMemberGroups`.
3. Cache memberships aggressively.
4. Apply security trimming after retrieval and before LLM access.
5. Optionally keep `User` and `Group` nodes for governance and organizational reasoning.

This follows the same architectural principles used by modern enterprise search engines while remaining compatible with graph-native retrieval and agentic navigation.


# 📋 Overview of Python Approaches for SharePoint

| Approach / Library | Best Used For | Pros | Cons |
|---|---|---|---|
| Microsoft Graph SDK (Official Microsoft Library) | Modern, cloud-first Microsoft 365 apps and general file/folder management. | • Officially supported by Microsoft. • Native, up-to-date OAuth2/Entra ID authentication. • Single endpoint for all M365 data (Teams, Outlook, OneDrive). | • Lacks deep, specialized SharePoint coverage (e.g., advanced list features). • Does not support SharePoint On-Premise. |
| Office365-REST-Python-Client (Community Standard) | Advanced SharePoint Online workflows, heavy list manipulation, and tenant administration. | • Full feature parity with the SharePoint REST API. • Handles complex SharePoint-specific objects easily. • Highly mature with massive community usage. | • Community-driven (not officially supported by Microsoft). • Can have a slightly steeper learning curve than Graph. |
| O365 & sharepoint-api-py (Wrapper / Async) | Rapid prototyping or modern, asynchronous AI data pipelines. | • O365 offers a clean, highly Pythonic syntax. • sharepoint-api-py features async/streaming for fast file handling. | • Third-party wrappers dependent on Graph API updates. • Smaller community support compared to the main libraries. |
| SharePlum (Legacy) | Connecting to older, local corporate networks. | • The only reliable choice for SharePoint On-Premise (2013–2019). • Supports older NTLM/Domain authentication. | • No longer actively updated for modern SharePoint Online features. • Restricted mostly to basic list and file operations. |

------------------------------
## 📐 Recommended Web Architecture
If your goal is to build a user-facing application, use a split architecture:

   1. Backend (Python): Use FastAPI or Flask to handle processing logic, using either the Microsoft Graph SDK (for clean file management) or Office365-REST-Python-Client (for detailed SharePoint data).
   2. Frontend (SharePoint): Deploy custom UI elements directly into your intranet using SPFx (SharePoint Framework) with React, which routes data requests securely back to your Python backend.

------------------------------
Which approach aligns better with your setup? Let me know if you want to see a Graph SDK file upload snippet or if you need help choosing between the Graph API vs. SharePoint REST API permissions.

