"""
knowledge_base.py - Build and load the phishing-domain FAISS knowledge base.

Documents are fetched from NIST and APWG using a simple web crawler,
then chunked (512 tokens, 64 overlap) and embedded with
BAAI/bge-large-en-v1.5 before being stored in a local FAISS index.
"""
import json
import os
import pickle
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

KB_DIR   = Path(__file__).parent.parent / "knowledge_base"
IDX_PATH = KB_DIR / "faiss_index"
DOC_PATH = KB_DIR / "documents.pkl"
META_PATH = KB_DIR / "metadata.json"
RAW_DIR = KB_DIR / "raw"
PROCESSED_DIR = KB_DIR / "processed"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CATALOG_VERSION = "full-doc-three-sources-v1"
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256
APPROX_TOKEN_CHUNK_SIZE = 512
APPROX_TOKEN_CHUNK_OVERLAP = 64

# ── Phishing-domain seed documents ──────────────────────────────────────────
AUTHORITATIVE_SOURCES = [
    {
        "id": "nist-sp-800-61r3",
        "title": "NIST SP 800-61 Rev. 3 Incident Response",
        "source": "NIST CSRC",
        "authority": "NIST",
        "category": "Incident Response",
        "doc_type": "standard",
        "url": "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        "updated": "2025-04-03",
        "status": "Indexed",
        "summary": (
            "Current NIST incident response recommendations aligned to CSF 2.0."
        ),
        "content": """
NIST SP 800-61 Rev. 3 frames incident response as part of cybersecurity risk
management and the NIST Cybersecurity Framework 2.0 functions. Phishing events
should be handled through preparation, detection, analysis, response,
communications, recovery, and improvement activities rather than as isolated
helpdesk tickets.

For phishing incidents, organizations should define reporting channels, triage
criteria, evidence collection procedures, escalation paths, containment
actions, eradication steps, and recovery validation. Relevant evidence includes
message headers, sender infrastructure, URLs, attachments, endpoint telemetry,
identity-provider logs, mail gateway logs, and user reports.

Effective response reduces the likelihood and impact of incidents by
coordinating prevention, detection, containment, recovery, and lessons learned.
Post-incident activities should update playbooks, detection rules, awareness
material, and risk registers.
""",
    },
    {
        "id": "nist-sp-800-177r1",
        "title": "NIST SP 800-177 Email Authentication Guidelines",
        "source": "NIST CSRC",
        "authority": "NIST",
        "category": "Email Security",
        "doc_type": "standard",
        "url": "https://csrc.nist.gov/pubs/sp/800/177/r1/final",
        "updated": "2019-02-26",
        "status": "Indexed",
        "summary": (
            "Trustworthy email guidance covering SPF, DKIM, DMARC, TLS, "
            "DANE, and S/MIME."
        ),
        "content": """
NIST SP 800-177 Rev. 1 gives recommendations for enhancing trust in email for
enterprise administrators and security teams. It recommends sender-domain
authentication mechanisms including SPF, DKIM, and DMARC, and also discusses
transport and content protections such as TLS, DANE, and S/MIME.

Key technical controls:
  1. SPF (Sender Policy Framework) – publishes authorised mail-server IPs
     in DNS to prevent domain spoofing.
  2. DKIM (DomainKeys Identified Mail) – cryptographically signs outgoing
     messages so recipients can verify authenticity.
  3. DMARC (Domain-based Message Authentication, Reporting and Conformance)
     – builds on SPF/DKIM to specify how receivers should handle failures
     and enables aggregate/forensic reporting.

Deployment recommendations:
  • Start DMARC in 'p=none' (monitor) mode, then escalate to 'quarantine'
    and finally 'reject'.
  • Require TLS encryption for SMTP relay connections.
  • Monitor DMARC aggregate reports weekly for anomalies.
  • Apply these controls to all organisational domains, including parked
    and sub-domains that never send mail.
""",
    },
    {
        "id": "cisa-counter-phishing",
        "title": "CISA Counter-Phishing Recommendations",
        "source": "CISA",
        "authority": "CISA",
        "category": "Phishing Tactics",
        "doc_type": "guidance",
        "url": (
            "https://www.cisa.gov/resources-tools/resources/"
            "capacity-enhancement-guide-federal-agencies-counter-phishing"
        ),
        "updated": "2024-10-01",
        "status": "Indexed",
        "summary": (
            "Technical controls for secure email gateways, web protections, "
            "endpoint hardening, and endpoint protections."
        ),
        "content": """
CISA's counter-phishing recommendations group technical capabilities into four
areas: secure email gateway controls, outbound web-browsing protections,
hardened user endpoints, and endpoint protections. These controls complement,
but do not replace, user awareness and reporting programs.

Secure email gateway controls should detect spoofing, malicious attachments,
dangerous links, and suspicious sender patterns. Web protections should inspect
or block known malicious destinations and risky redirects. Endpoint hardening
should reduce the impact of malicious attachments, scripts, macros, and browser
abuse. Endpoint protections should correlate suspicious email activity with
file execution, process creation, network connections, and credential theft.
""",
    },
    {
        "id": "mitre-attack-t1566",
        "title": "MITRE ATT&CK T1566 Phishing",
        "source": "MITRE ATT&CK",
        "authority": "MITRE",
        "category": "Phishing Tactics",
        "doc_type": "technique",
        "url": "https://attack.mitre.org/techniques/T1566/",
        "updated": "2025-10-24",
        "status": "Indexed",
        "summary": (
            "Enterprise ATT&CK phishing technique with attachment, link, "
            "service, and voice sub-techniques."
        ),
        "content": """
MITRE ATT&CK technique T1566 describes phishing as electronically delivered
social engineering used for initial access. Sub-techniques include
spearphishing attachment, spearphishing link, spearphishing via service, and
spearphishing voice.

Adversaries may send malicious links or attachments, abuse cloud and SaaS
services, spoof senders, manipulate email metadata, hijack legitimate threads,
or direct victims to call a phone number. Detection should correlate email
delivery, link clicks, attachment saves, process execution, identity-provider
events, suspicious OAuth grants, and anomalous login behavior.

Mitigations include anti-malware, network intrusion prevention, web-content
restrictions, email authentication such as SPF and DKIM, user training, and
controls that connect mail telemetry with endpoint and identity telemetry.
""",
    },
    {
        "id": "fbi-bec",
        "title": "FBI Business Email Compromise Guidance",
        "source": "FBI",
        "authority": "FBI",
        "category": "BEC",
        "doc_type": "guidance",
        "url": (
            "https://www.fbi.gov/how-we-can-help-you/safety-resources/"
            "scams-and-safety/common-scams-and-crimes/"
            "business-email-compromise"
        ),
        "updated": "2025-11-01",
        "status": "Indexed",
        "summary": (
            "BEC scam patterns, spoofing, spearphishing, malware use, and "
            "payment verification advice."
        ),
        "content": """
The FBI describes business email compromise as a financially damaging online
crime that exploits reliance on email for personal and professional business.
BEC messages often appear to come from a known source making a legitimate
request, such as a vendor invoice change, executive gift-card purchase, or real
estate wire-transfer instruction.

Attackers may spoof email accounts or websites, send spearphishing messages,
or use malware to gain access to legitimate email threads and billing details.
Protective measures include careful examination of email addresses, URLs, and
spelling; avoiding unsolicited links and attachments; enabling MFA; verifying
payment and purchase requests using an independent channel; and treating urgent
requests with additional scrutiny.
""",
    },
    {
        "id": "cisa-recognize-report-phishing",
        "title": "CISA Recognize and Report Phishing",
        "source": "CISA",
        "authority": "CISA",
        "category": "Awareness",
        "doc_type": "guidance",
        "url": "https://www.cisa.gov/secure-our-world/recognize-and-report-phishing",
        "updated": "2025-01-01",
        "status": "Indexed",
        "summary": (
            "User-facing phishing recognition and reporting guidance."
        ),
        "content": """
CISA advises users to recognize phishing messages by looking for urgent or
emotionally appealing language, requests for personal or financial
information, untrusted shortened URLs, unexpected attachments, impersonation,
and pressure to act quickly. Phishing messages can arrive through email, text,
direct message, social media, or voice channels.

Organizations should make reporting easy, triage user reports quickly, and use
reported messages to tune mail filters, block malicious domains, update
awareness content, and identify affected users or endpoints.
""",
    },
    {
        "id": "cisa-mfa",
        "title": "CISA Phishing-Resistant MFA",
        "source": "CISA",
        "authority": "CISA",
        "category": "Identity Security",
        "doc_type": "guidance",
        "url": "https://www.cisa.gov/mfa",
        "updated": "2025-09-01",
        "status": "Indexed",
        "summary": (
            "MFA hierarchy and phishing-resistant FIDO/WebAuthn guidance."
        ),
        "content": """
CISA recommends MFA because compromised passwords are commonly harvested
through phishing. Not all MFA methods provide the same protection. CISA
identifies phishing-resistant MFA as the standard organizations should strive
for, with FIDO/WebAuthn as a widely available phishing-resistant option.

FIDO/WebAuthn can block attempts where a user is tricked into logging into a
fake website because authentication is cryptographically bound to the legitimate
site. Where phishing-resistant MFA cannot be deployed immediately, number
matching can reduce push-bombing and SMS-based attack risk, but it is an
interim control rather than the target state.
""",
    },
    {
        "id": "microsoft-bec-playbook",
        "title": "Microsoft BEC Detection and Response Practices",
        "source": "Microsoft Security",
        "authority": "Microsoft",
        "category": "BEC",
        "doc_type": "vendor guidance",
        "url": (
            "https://www.microsoft.com/en-us/security/business/security-101/"
            "what-is-business-email-compromise-bec"
        ),
        "updated": "2025-01-01",
        "status": "Indexed",
        "summary": (
            "Operational BEC patterns, identity controls, mailbox rules, and "
            "payment-flow safeguards."
        ),
        "content": """
Business email compromise detection should combine mailbox, identity, endpoint,
and payment-process signals. Useful signals include suspicious inbox rules,
external forwarding, impossible travel, risky sign-ins, new OAuth grants,
unusual sender display names, lookalike domains, and messages that ask staff to
bypass normal payment workflows.

Controls include phishing-resistant MFA for privileged and finance users,
conditional access, mailbox auditing, external sender labeling, domain
impersonation protection, safe links and attachments, anomaly detection on
payment requests, and out-of-band approval for invoice or bank-account changes.
""",
    },
    {
        "id": "dmarc-org-overview",
        "title": "DMARC Domain Authentication Overview",
        "source": "DMARC.org",
        "authority": "DMARC.org",
        "category": "Email Security",
        "doc_type": "standard guidance",
        "url": "https://dmarc.org/",
        "updated": "2025-01-01",
        "status": "Indexed",
        "summary": (
            "DMARC policy alignment, reporting, quarantine, and reject "
            "deployment concepts."
        ),
        "content": """
DMARC builds on SPF and DKIM by allowing domain owners to publish a policy for
messages that fail authentication and alignment checks. Policies commonly move
from monitoring to quarantine and then reject as organizations gain confidence
in legitimate mail flows.

Aggregate DMARC reports help domain owners discover legitimate senders,
unauthorized senders, forwarding behavior, and spoofing campaigns. Effective
deployment covers active, parked, and lookalike domains, and should be paired
with DKIM signing, SPF record hygiene, monitoring, and clear ownership of mail
streams.
""",
    },
    {
        "id": "apwg-trends",
        "title": "APWG Phishing Activity Trends",
        "source": "APWG",
        "authority": "APWG",
        "category": "Threat Intelligence",
        "doc_type": "report",
        "url": "https://apwg.org/trendsreports/",
        "updated": "2025-01-01",
        "status": "Indexed",
        "summary": (
            "Phishing activity trends, target sectors, credential theft, and "
            "campaign observations."
        ),
        "content": """
APWG phishing activity trend reports provide measurements of observed phishing
activity, affected sectors, phishing sites, credential-theft campaigns, and
changes in attacker infrastructure. Trend data is useful for prioritizing
controls, selecting awareness themes, and explaining why technical defenses and
rapid reporting remain necessary.

Threat intelligence from phishing reports should be normalized into indicators
such as URLs, domains, hosting providers, brands impersonated, attachment
types, sender infrastructure, and campaign themes. These indicators can feed
mail gateway blocks, DNS filtering, browser isolation, takedown workflows, and
SOC hunting.
""",
    },
    {
        "id": "email-header-analysis",
        "title": "Email Header Analysis Techniques",
        "source": "PhishingGuard Analyst Notes",
        "authority": "Operational Playbook",
        "category": "Email Security",
        "doc_type": "playbook",
        "url": "local://email-header-analysis",
        "updated": "2026-04-23",
        "status": "Indexed",
        "summary": (
            "Practical header triage workflow for sender authentication and "
            "routing anomalies."
        ),
        "content": """
Email header analysis should inspect From, Reply-To, Return-Path,
Authentication-Results, Received chains, Message-ID, DKIM-Signature, ARC
headers, SPF results, DKIM results, DMARC results, sending IP reputation, and
domain age. Analysts should compare display-name identity with authenticated
domain identity and look for mismatches between claimed sender, envelope
sender, and link destinations.

High-risk findings include failed DMARC alignment, newly registered domains,
lookalike domains, reply-to mismatches, unexpected forwarding paths, forged
internal display names, links hidden behind redirects, and attachments that
launch child processes or request macro execution.
""",
    },
    {
        "id": "spear-phishing-response-playbook",
        "title": "Spear Phishing Indicators and Response Protocols",
        "source": "PhishingGuard Analyst Notes",
        "authority": "Operational Playbook",
        "category": "Phishing Tactics",
        "doc_type": "playbook",
        "url": "local://spear-phishing-response",
        "updated": "2026-04-23",
        "status": "Indexed",
        "summary": (
            "SOC playbook for targeted phishing detection, containment, and "
            "post-incident improvement."
        ),
        "content": """
Spear phishing attacks are personalized and may bypass generic mail filters by
referencing real projects, colleagues, vendors, invoices, shared documents, or
executive names.

Detection strategies:
  1. Analyse email headers for source IP mismatches.
  2. Verify sender domain against known-good lists.
  3. Use Natural Language Processing (NLP) to flag urgency cues.
  4. Deploy sandbox analysis for attachments and links.
  5. Cross-reference URLs against phishing databases (PhishTank, APWG).

Incident response steps (NIST SP 800-61):
  Phase 1 – Preparation: maintain an IR plan with phishing-specific playbook.
  Phase 2 – Detection and Analysis: correlate email logs, identify IOCs.
  Phase 3 – Containment: isolate compromised endpoint, reset credentials,
             block malicious domains at the gateway.
  Phase 4 – Eradication and Recovery: remove malware, restore from backup.
  Phase 5 – Post-Incident: update blocklists, retrain affected employees.

Escalation: notify SOC within one hour of confirmed phishing; escalate to
CISO if executive accounts or sensitive data are involved.
""",
    },
    {
        "id": "url-obfuscation-iocs",
        "title": "URL Obfuscation Methods in Phishing Campaigns",
        "source": "PhishingGuard Analyst Notes",
        "authority": "Operational Playbook",
        "category": "Phishing Tactics",
        "doc_type": "playbook",
        "url": "local://url-obfuscation-iocs",
        "updated": "2026-04-23",
        "status": "Indexed",
        "summary": (
            "IOC taxonomy for malicious links, redirects, and attachment "
            "delivery patterns."
        ),
        "content": """
Email-based IOCs:
  • Sender domain registered < 30 days ago.
  • Reply-to address differs from the From address.
  • Display name spoofing (e.g. 'Microsoft Support <attacker@evil.com>').
  • Excessive urgency, threats, or promises in the subject line.
  • Misspellings or homoglyph substitutions in domain names
    (e.g. 'rn' rendering as 'm').

URL-based IOCs:
  • URL shorteners masking the true destination.
  • IP-address-based URLs (e.g. http://192.168.x.x/login).
  • Excessive subdomains (e.g. secure.login.verify.paypal.evil.com).
  • Mismatched anchor text vs href attribute.
  • TLS certificate for a newly registered or free domain.

Attachment-based IOCs:
  • Office files with macros enabled by default.
  • Password-protected archives (bypass email scanning).
  • Double file extensions (e.g. document.pdf.exe).
  • ISO or IMG files (mount as virtual drives, bypass Mark-of-the-Web).

Behavioural IOCs:
  • Unusual after-hours access or login from new geolocation.
  • Mass data exfiltration shortly after account compromise.
  • New email forwarding rules created silently.
""",
    },
]

AUTHORITATIVE_SOURCES = [
    {
        "id": "nist-sp-800-177r1",
        "title": "NIST SP 800-177 Rev. 1 Trustworthy Email",
        "source": "NIST CSRC",
        "authority": "NIST",
        "category": "Email Security",
        "doc_type": "pdf",
        "fetch_mode": "pdf",
        "url": "https://csrc.nist.gov/pubs/sp/800/177/r1/final",
        "download_url": (
            "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/"
            "NIST.SP.800-177r1.pdf"
        ),
        "updated": "2019-02-26",
        "status": "Indexed",
        "summary": (
            "Full NIST guidance for trustworthy email, including SPF, DKIM, "
            "DMARC, TLS, DANE, DNSSEC, S/MIME, and operational deployment."
        ),
        "content": "",
    },
    {
        "id": "apwg-phishing-trends-q3-2024",
        "title": "APWG Phishing Activity Trends Report Q3 2024",
        "source": "APWG",
        "authority": "APWG",
        "category": "Threat Intelligence",
        "doc_type": "pdf",
        "fetch_mode": "pdf",
        "url": "https://apwg.org/trendsreports/",
        "download_url": (
            "https://docs.apwg.org/reports/apwg_trends_report_q3_2024.pdf"
        ),
        "updated": "2024-12-04",
        "status": "Indexed",
        "summary": (
            "Full APWG quarterly phishing trends report covering phishing "
            "volume, sector targeting, domain abuse, and contributor findings."
        ),
        "content": "",
    },
    {
        "id": "stackoverflow-phishing-crawl",
        "title": "Stack Overflow Phishing Discussions Crawl",
        "source": "Stack Overflow",
        "authority": "Community Q&A",
        "category": "Developer Signals",
        "doc_type": "web crawl",
        "fetch_mode": "stackoverflow",
        "url": "https://stackoverflow.com/search?q=phishing",
        "download_url": (
            "https://api.stackexchange.com/2.3/search/advanced"
        ),
        "updated": "2026-04-23",
        "status": "Indexed",
        "summary": (
            "Crawler-collected Stack Overflow questions and answers related "
            "to phishing, suspicious links, email handling, and web security."
        ),
        "content": "",
    },
]

PHISHING_KNOWLEDGE = AUTHORITATIVE_SOURCES


def _fetch_url_text(url: str, timeout: int = 10) -> str:
    """Fetch and extract plain text from a URL (best-effort)."""
    if url.startswith("local://"):
        return ""
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "PhishingGuard-RAG/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:8000]
    except Exception:
        return ""


def _clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text.strip()


def _safe_filename(source_id: str, suffix: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", source_id).strip("-")
    return RAW_DIR / f"{safe_id}{suffix}"


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Full-document PDF ingestion requires `pypdf`. Install project "
            "dependencies with `pip install -r requirements.txt`, then "
            "rebuild the knowledge base."
        ) from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = _clean_text(page_text)
        if page_text:
            pages.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def _fetch_pdf_document(item: dict) -> str:
    url = item.get("download_url") or item["url"]
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "PhishingGuard-RAG/1.0"},
    )
    response.raise_for_status()

    raw_path = _safe_filename(item["id"], ".pdf")
    raw_path.write_bytes(response.content)
    return _extract_pdf_text(response.content)


def _fetch_stackoverflow_document(item: dict) -> str:
    queries = [
        q.strip() for q in os.getenv(
            "STACKEXCHANGE_CRAWL_QUERIES",
            "phishing;email spoofing;DMARC;SPF DKIM;malicious link;"
            "suspicious email;phishing email;url spoofing;open redirect phishing",
        ).split(";")
        if q.strip()
    ]
    sites = [
        s.strip() for s in os.getenv(
            "STACKEXCHANGE_CRAWL_SITES",
            "stackoverflow;security;serverfault",
        ).split(";")
        if s.strip()
    ]
    pagesize = int(os.getenv("STACKEXCHANGE_CRAWL_PAGESIZE", "20"))
    max_questions = int(os.getenv("STACKEXCHANGE_CRAWL_MAX_QUESTIONS", "60"))

    search_payloads = []
    questions_by_key = {}
    for site in sites:
        for query in queries:
            response = requests.get(
                item["download_url"],
                params={
                    "order": "desc",
                    "sort": "relevance",
                    "q": query,
                    "site": site,
                    "pagesize": pagesize,
                    "filter": "withbody",
                },
                timeout=30,
                headers={"User-Agent": "PhishingGuard-RAG/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            payload["_crawler_site"] = site
            payload["_crawler_query"] = query
            search_payloads.append(payload)
            for question in payload.get("items", []):
                key = (site, question.get("question_id"))
                if question.get("question_id") and key not in questions_by_key:
                    question["_crawler_site"] = site
                    question["_crawler_query"] = query
                    questions_by_key[key] = question

    questions = sorted(
        questions_by_key.values(),
        key=lambda q: (q.get("score", 0), q.get("answer_count", 0)),
        reverse=True,
    )[:max_questions]
    answers_by_question = {}
    for site in sites:
        site_question_ids = [
            str(q.get("question_id")) for q in questions
            if q.get("_crawler_site") == site and q.get("question_id")
        ]
        if not site_question_ids:
            continue
        answers_url = (
            "https://api.stackexchange.com/2.3/questions/"
            f"{';'.join(site_question_ids)}/answers"
        )
        answer_resp = requests.get(
            answers_url,
            params={
                "order": "desc",
                "sort": "votes",
                "site": site,
                "pagesize": min(100, len(site_question_ids) * 3),
                "filter": "withbody",
            },
            timeout=30,
            headers={"User-Agent": "PhishingGuard-RAG/1.0"},
        )
        answer_resp.raise_for_status()
        for answer in answer_resp.json().get("items", []):
            qid = answer.get("question_id")
            key = f"{site}:{qid}"
            answer["_crawler_site"] = site
            answers_by_question.setdefault(key, []).append(answer)

    sections = []
    for question in questions:
        qid = question.get("question_id")
        title = _clean_text(question.get("title", "Untitled"))
        link = question.get("link", "")
        body = _clean_text(question.get("body", ""))
        site = question.get("_crawler_site", "stackoverflow")
        query = question.get("_crawler_query", "")
        answer_key = f"{site}:{qid}"
        sections.append(
            f"Site: {site}\nMatched Query: {query}\nQuestion: {title}\n"
            f"URL: {link}\nScore: {question.get('score', 0)}\n"
            f"Answers: {question.get('answer_count', 0)}\n\n{body}"
        )
        for i, answer in enumerate(answers_by_question.get(answer_key, [])[:3], 1):
            answer_body = _clean_text(answer.get("body", ""))
            sections.append(
                f"Answer {i} for {site} question {qid}\nScore: "
                f"{answer.get('score', 0)}\n\n{answer_body}"
            )

    raw_path = _safe_filename(item["id"], ".json")
    raw_path.write_text(json.dumps({
        "crawler_config": {
            "queries": queries,
            "sites": sites,
            "pagesize": pagesize,
            "max_questions": max_questions,
        },
        "search_payloads": search_payloads,
        "questions": questions,
        "answers_by_question": answers_by_question,
    }, indent=2), encoding="utf-8")
    return "\n\n---\n\n".join(sections)


def _load_full_document(item: dict) -> str:
    fetch_mode = item.get("fetch_mode", "html")
    if fetch_mode == "pdf":
        return _fetch_pdf_document(item)
    if fetch_mode == "stackoverflow":
        return _fetch_stackoverflow_document(item)
    fetched_text = _fetch_url_text(item["url"])
    return fetched_text or item.get("content", "")


def _catalog_needs_rebuild() -> bool:
    if not IDX_PATH.exists() or not DOC_PATH.exists() or not META_PATH.exists():
        return True
    try:
        metadata = json.loads(META_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return metadata.get("catalog_version") != CATALOG_VERSION


def _source_metadata(item: dict) -> dict:
    return {
        "source_id": item["id"],
        "source": item["source"],
        "title": item["title"],
        "authority": item["authority"],
        "category": item["category"],
        "doc_type": item["doc_type"],
        "url": item["url"],
        "updated": item["updated"],
        "status": item["status"],
        "summary": item["summary"],
    }


def get_source_catalog() -> list:
    """Return the configured authoritative source catalog for the UI."""
    return [dict(item) for item in AUTHORITATIVE_SOURCES]


def get_index_metadata() -> dict:
    """Read build metadata for the current FAISS index."""
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def get_kb_document_stats() -> dict:
    """Return document/chunk/vector statistics derived from the actual index."""
    docs = []
    if DOC_PATH.exists():
        try:
            with open(DOC_PATH, "rb") as f:
                docs = pickle.load(f)
        except Exception:
            docs = []

    source_ids = {
        d.metadata.get("source_id") for d in docs
        if getattr(d, "metadata", None) and d.metadata.get("source_id")
    }
    metadata = get_index_metadata()
    chunks = len(docs)
    return {
        "documents": len(source_ids) or len(AUTHORITATIVE_SOURCES),
        "chunks": chunks,
        "embeddings": chunks,
        "status": "Active" if IDX_PATH.exists() and chunks else "Not built",
        "last_sync": metadata.get("built_at", "Not synced"),
        "catalog_version": metadata.get("catalog_version", CATALOG_VERSION),
        "embedding_model": metadata.get("embedding_model", EMBEDDING_MODEL),
        "chunk_size": metadata.get("chunk_size", CHUNK_SIZE),
        "chunk_overlap": metadata.get("chunk_overlap", CHUNK_OVERLAP),
        "approx_token_chunk_size": metadata.get(
            "approx_token_chunk_size", APPROX_TOKEN_CHUNK_SIZE
        ),
        "approx_token_chunk_overlap": metadata.get(
            "approx_token_chunk_overlap", APPROX_TOKEN_CHUNK_OVERLAP
        ),
    }


def build_knowledge_base(force_rebuild: bool = False):
    """
    Build the FAISS knowledge base from authoritative source documents.
    If the index already exists, skip unless force_rebuild=True.
    """
    KB_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and not _catalog_needs_rebuild():
        return

    raw_docs = []
    source_text_lengths = {}
    for item in AUTHORITATIVE_SOURCES:
        content = _load_full_document(item)
        if len(content.strip()) < 1200:
            raise RuntimeError(
                f"Full-document ingestion for {item['title']} produced only "
                f"{len(content.strip())} characters. Check network access, "
                "PDF parsing dependencies, or crawler configuration."
            )
        processed_path = PROCESSED_DIR / f"{item['id']}.txt"
        processed_path.write_text(content, encoding="utf-8")
        source_text_lengths[item["id"]] = len(content)
        raw_docs.append(Document(
            page_content=content.strip(),
            metadata={
                **_source_metadata(item),
                "processed_path": str(processed_path),
                "text_length": len(content),
            },
        ))

    # Chunk: 512 tokens ≈ 2048 chars for this model; overlap = 64 tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # Embed and index
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(IDX_PATH))

    with open(DOC_PATH, "wb") as f:
        pickle.dump(chunks, f)

    META_PATH.write_text(json.dumps({
        "catalog_version": CATALOG_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL,
        "documents": len(AUTHORITATIVE_SOURCES),
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "approx_token_chunk_size": APPROX_TOKEN_CHUNK_SIZE,
        "approx_token_chunk_overlap": APPROX_TOKEN_CHUNK_OVERLAP,
        "source_text_lengths": source_text_lengths,
    }, indent=2))


def load_knowledge_base():
    """Load the FAISS index from disk (build it first if missing)."""
    build_knowledge_base()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(IDX_PATH), embeddings, allow_dangerous_deserialization=True
    )
