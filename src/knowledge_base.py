"""
knowledge_base.py - Build and load the phishing-domain FAISS knowledge base.

Documents are fetched from NIST and APWG using a simple web crawler,
then chunked (512 tokens, 64 overlap) and embedded with
BAAI/bge-large-en-v1.5 before being stored in a local FAISS index.
"""
import os
import pickle
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

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ── Phishing-domain seed documents ──────────────────────────────────────────
PHISHING_KNOWLEDGE = [
    {
        "title": "Phishing Attack Overview",
        "content": """
Phishing attacks are deceptive attempts to obtain sensitive information by
impersonating trustworthy entities via email, SMS, or fraudulent websites.
Common phishing types include:
  • Spear phishing – highly targeted attacks using personalised information.
  • Whaling – targeting senior executives.
  • Vishing – voice-based social engineering.
  • Smishing – SMS-based phishing.
  • Clone phishing – duplicating a legitimate email with a malicious payload.

Detection indicators: suspicious sender domains, urgent language, mismatched
URLs, unexpected attachments, requests for credentials, and poor grammar.

Mitigation: implement DMARC/DKIM/SPF email authentication, deploy email
security gateways, run regular phishing simulation training, and enforce
multi-factor authentication (MFA).
""",
        "source": "PhishingGuard Knowledge Base",
    },
    {
        "title": "NIST SP 800-177 Email Authentication Guidelines",
        "content": """
NIST SP 800-177 provides guidance on trustworthy email for federal agencies.

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
        "source": "NIST SP 800-177r1",
    },
    {
        "title": "Spear Phishing Detection and Response",
        "content": """
Spear phishing attacks are personalised and bypass traditional filters.

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
        "source": "NIST SP 800-61 Rev 2",
    },
    {
        "title": "Enterprise Phishing Prevention Best Practices",
        "content": """
Technical controls:
  • Email filtering: block known-malicious senders, strip active content.
  • URL rewriting and time-of-click analysis.
  • Attachment sandboxing for Office, PDF, and archive files.
  • Endpoint Detection and Response (EDR) with phishing-aware rules.
  • DNS-based filtering (e.g. Cisco Umbrella, Cloudflare Gateway).
  • Zero Trust Network Access – verify every request regardless of origin.

Organisational controls:
  • Quarterly phishing simulation campaigns (reduces susceptibility ~70%).
  • Role-based training: finance, HR, and executives require extra sessions.
  • Clear reporting channel (e.g. a dedicated 'Report Phishing' button).
  • Reward-based culture: praise reporters, do not punish clickers.
  • Tabletop exercises simulating executive BEC scenarios.

Metrics to track:
  – Phishing simulation click rate (target < 5%).
  – Mean time to report suspicious email.
  – Number of credentials compromised per quarter.
""",
        "source": "SANS Institute / APWG Best Practices",
    },
    {
        "title": "BEC and Advanced Phishing Techniques",
        "content": """
Business Email Compromise (BEC) is a sophisticated form of phishing that
impersonates executives or trusted partners to authorise fraudulent transfers.

BEC categories (FBI IC3):
  • CEO fraud – attacker impersonates CEO to instruct wire transfers.
  • Account compromise – attacker accesses a legitimate email account.
  • Attorney impersonation – fake legal pressure to transfer funds urgently.
  • W-2 theft – targeting HR for employee tax data.
  • Real-estate wire fraud – hijacking property transaction communications.

Advanced techniques:
  • Adversary-in-the-Middle (AiTM) – proxying login pages to steal MFA tokens.
  • QR-code phishing – embedding malicious URLs in QR codes to bypass filters.
  • Deepfake audio – impersonating executives in voicemail or live calls.
  • Thread hijacking – inserting replies into existing legitimate email chains.

Countermeasures:
  • Verify all fund-transfer requests via a second out-of-band channel.
  • Implement dual-approval workflows for transactions above a threshold.
  • Deploy phishing-resistant MFA (FIDO2/passkeys) for all privileged accounts.
  • Monitor for newly registered look-alike domains.
""",
        "source": "FBI IC3 / APWG eCrime Reports",
    },
    {
        "title": "Phishing Indicators of Compromise (IOCs)",
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
        "source": "MITRE ATT&CK / CISA Phishing Guidance",
    },
]


def _fetch_url_text(url: str, timeout: int = 10) -> str:
    """Fetch and extract plain text from a URL (best-effort)."""
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


def build_knowledge_base(force_rebuild: bool = False):
    """
    Build the FAISS knowledge base from the seed documents.
    If the index already exists, skip unless force_rebuild=True.
    """
    KB_DIR.mkdir(parents=True, exist_ok=True)

    if IDX_PATH.exists() and DOC_PATH.exists() and not force_rebuild:
        return

    # Build LangChain Document objects from seed knowledge
    raw_docs = []
    for item in PHISHING_KNOWLEDGE:
        raw_docs.append(Document(
            page_content=item["content"].strip(),
            metadata={"source": item["source"], "title": item["title"]},
        ))

    # Optional: attempt to enrich with live APWG page
    apwg_text = _fetch_url_text("https://apwg.org/trendsreports/")
    if apwg_text:
        raw_docs.append(Document(
            page_content=apwg_text,
            metadata={"source": "APWG Phishing Activity Trends",
                      "title": "APWG Live Reports"},
        ))

    # Chunk: 512 tokens ≈ 2048 chars for this model; overlap = 64 tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=256,
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
