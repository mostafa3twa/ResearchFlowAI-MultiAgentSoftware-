"""
ResearchFlow AI - Multi-Agent System
Agent 1: Search Queries Recommendation Agent  (CrewAI + LLM)
Agent 2: Search Engine                        (Pure Python + Tavily - NO LLM!)
Agent 3: Web Scraper                          (Pure Python + ScrapGraph - NO LLM overhead!)
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import agentops
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import os
import json
import re
import sys
from tavily import TavilyClient
from scrapegraph_py import ScrapeGraphAI as ScrapGraphClient
from scrapegraph_py.schemas import FetchConfig

# Force UTF-8 output on Windows — prevents emoji UnicodeEncodeError from libraries like AgentOps
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables
load_dotenv()

# Configuration
class Config:
    """Application configuration"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq API key for Agent 1 only
    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    SCRAPEGRAPH_API_KEY = os.getenv("SCRAPEGRAPH_API_KEY")
    OUTPUT_DIR = "./ai-agent-output"
    NO_KEYWORDS = 10       # Max queries Agent 1 generates
    MAX_RESULTS = 5        # Max Tavily results per query
    SCORE_THRESHOLD = 0.50 # Minimum relevance score to keep a result
    TOP_SCRAPE = 10        # Max URLs Agent 3 scrapes

# Validate API keys
def validate_config():
    """Validate that all required API keys are present"""
    required_keys = {
        "GROQ_API_KEY": Config.GROQ_API_KEY,
        "AGENTOPS_API_KEY": Config.AGENTOPS_API_KEY,
        "TAVILY_API_KEY": Config.TAVILY_API_KEY,
        "SCRAPEGRAPH_API_KEY": Config.SCRAPEGRAPH_API_KEY,
    }

    missing_keys = [key for key, value in required_keys.items() if not value]

    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

    print("\u2713 All required API keys are configured")
    print(f"\u2713 Agent 1 (LLM) API Key: {Config.GROQ_API_KEY[:20]}...")
    print("\u2713 Agent 2 (Python) uses Tavily only - NO LLM needed!")
    print("\u2713 Agent 3 (Python) uses ScrapGraph only - NO LLM overhead!")

# Initialize environment
def initialize_environment():
    """Initialize the AI agents environment"""
    
    # # Set environment variables for Groq
    # if Config.GROQ_API_KEY:
    #     os.environ["GROQ_API_KEY"] = Config.GROQ_API_KEY
    
    # Initialize AgentOps
    agentops.init(
        api_key=Config.AGENTOPS_API_KEY,
        skip_auto_end_session=True,
        default_tags=['crewai', 'researchflow']
    )
    
    print(f"AgentOps endpoint: {agentops.get_client().config.exporter_endpoint}")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory created: {Config.OUTPUT_DIR}")

# Initialize clients
def initialize_clients():
    """Initialize API clients - LLM for Agent 1 only"""
    # Agent 1 needs LLM (creative task: generate varied search queries)
    llm_agent1 = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0,
        api_key=Config.GROQ_API_KEY,
    )
    print("\u2713 LLM for Agent 1 initialized")

    # Agent 2 needs only Tavily (mechanical task: call API + filter results)
    search_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
    print("\u2713 Tavily client for Agent 2 initialized (NO LLM!)")

    # Agent 3 needs only ScrapGraph (AI scraping on ScrapGraph's quota, NOT Groq!)
    scrape_client = ScrapGraphClient(api_key=Config.SCRAPEGRAPH_API_KEY)
    print("\u2713 ScrapGraph client for Agent 3 initialized (NO Groq LLM!)")

    return llm_agent1, search_client, scrape_client


# ============================================================
# PYDANTIC MODELS
# ============================================================

class SuggestedSearchQueries(BaseModel):
    """Model for search queries output from Agent 1"""
    queries: List[str] = Field(
        ...,
        title="Suggested search queries to be passed to the search engine",
        min_length=1,
        max_length=Config.NO_KEYWORDS,
        description="List of optimized search queries for product research"
    )


class SingleSearchResult(BaseModel):
    """Model for a single search result"""
    title: str = Field(..., description="Product title")
    url: str = Field(..., description="Product page URL")
    score: float = Field(..., description="Relevance score")
    search_query: str = Field(..., description="Original search query")


class AllSearchResults(BaseModel):
    """Model for all search results from Agent 2"""
    results: List[SingleSearchResult] = Field(
        ...,
        title="All search results from all queries",
        description="List of all search results collected from multiple search queries"
    )


# ============================================================
# PYDANTIC MODELS - AGENT 3
# ============================================================

class ProductSpec(BaseModel):
    """A single product specification key-value pair"""
    specification_name: str
    specification_value: str


class SingleExtractedProduct(BaseModel):
    """Model for a single scraped product from Agent 3"""
    page_url: str = Field(..., title="The original URL of the product page")
    product_title: str = Field(..., title="The title of the product")
    product_image_url: Optional[str] = Field(default=None, title="The URL of the product image")
    product_current_price: Optional[float] = Field(
        default=None,
        title="The current selling price as a plain number (e.g. 15999.0). null if not found on the page. NEVER return 0."
    )
    product_original_price: Optional[float] = Field(
        default=None,
        title="The original price before discount. null if no discount or not found."
    )
    product_discount_percentage: Optional[float] = Field(
        default=None,
        title="The discount percentage as a number (e.g. 15 for 15% off). null if not found."
    )
    product_specs: List[ProductSpec] = Field(
        ...,
        title="Key product specifications (focus on most important for comparison)",
        min_length=1,
        max_length=5,
    )


class AllExtractedProducts(BaseModel):
    """Model for all scraped products from Agent 3"""
    products: List[SingleExtractedProduct]


# Company Context
def create_company_context():
    """Create company context knowledge source"""
    about_company = (
        "Rankyx is a company that provides AI solutions to help websites "
        "refine their search and recommendation systems."
    )
    
    return StringKnowledgeSource(content=about_company)

# Agent 1: Search Queries Recommendation Agent
def create_search_queries_agent(llm):
    """Create the search queries recommendation agent"""
    
    agent = Agent(
        role="Search Queries Recommendation Agent",
        goal="\n".join([
            "To provide a list of suggested search queries to be passed to the search engine.",
            "The queries must be varied and looking for specific items."
        ]),
        backstory=(
            "The agent is designed to help in looking for products by providing "
            "a list of suggested search queries to be passed to the search engine "
            "based on the context provided."
        ),
        llm=llm,
        verbose=True,
    )
    
    return agent

def create_search_queries_task(agent):
    """Create the search queries recommendation task"""
    
    task = Task(
        description="\n".join([
            "Rankyx is looking to buy {product_name} at the best prices (value for a price strategy).",
            "The company targets these websites: {websites_list}",
            "The stores must sell the product in {country_name}.",
            "Generate at maximum {no_keywords} search queries.",
            "The search keywords must be in {language} language.",
            "",
            "IMPORTANT: Do NOT use the site: operator. It restricts results too much and reduces diversity.",
            "Instead, write natural purchase-intent queries that include:",
            "  - A specific brand + model name (e.g. De'Longhi Magnifica Evo, Nespresso Vertuo Next)",
            "  - A purchase signal (buy, price, offer, egypt, EGP)",
            "  - A soft site hint from the target websites list (e.g. 'amazon egypt', 'jumia egypt', 'noon egypt')",
            "",
            "Good examples:",
            "  De'Longhi Magnifica Evo espresso machine price egypt jumia",
            "  Nespresso Vertuo Next coffee machine buy amazon egypt EGP",
            "  Breville Barista Touch espresso machine noon egypt offer",
            "  Siemens EQ.9 coffee machine egypt price amazon",
            "  Philips 3200 LatteGo espresso machine jumia egypt",
            "",
            "Each query must target a specific product model, not a generic category.",
            "Vary the brands, models, and site hints across queries for maximum diversity.",
        ]),
        expected_output="A JSON object containing a list of suggested search queries.",
        output_json=SuggestedSearchQueries,
        output_file=os.path.join(Config.OUTPUT_DIR, "step_1_suggested_search_queries.json"),
        agent=agent
    )
    
    return task


# ============================================================
# AGENT 2: SEARCH ENGINE (Pure Python - NO LLM!)
# ============================================================

# Domains that are NEVER e-commerce product pages — block them before any other check
_BLOCKED_DOMAINS = {
    # Social media
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "tiktok.com", "youtube.com", "snapchat.com", "pinterest.com",
    "linkedin.com", "reddit.com",
    # Price-comparison / aggregator sites (not the seller)
    "pricena.com", "yaoota.com", "ksa-prices.com", "egprices.com",
    "compareit4me.com", "pricespy.com", "kanbkam.com", "pricepoint.com.eg",
    # Classifieds
    "olx.com.eg", "dubizzle.com", "hatla2ee.com",
    # Blogs / encyclopaedias / review sites
    "quora.com", "medium.com", "blogspot.com", "wordpress.com",
    "wikipedia.org", "foursquare.com", "yelp.com", "tripadvisor.com",
    "gsmarena.com", "notebookcheck.net",
}

# Patterns that indicate a search/category/listing page (not a single product)
_NON_PRODUCT_PATTERNS = [
    "/s?",          # Amazon search: /s?k=...
    "?k=",          # Amazon keyword search
    "/search",      # Generic search pages
    "search?",
    "search/",
    "/category",    # Category pages
    "/categories",
    "/collections", # Shopify collections
    "/marketplace", # Facebook / general marketplace
    "/stores/",     # Amazon store pages
    "/shop/",       # Generic shop pages
    "/browse",      # Browse pages
    "/listing",     # Listing pages
    "srsltid=",     # Google Shopping referral → usually listing pages
    "mlp-",         # Jumia multi-listing pages (/mlp-...)
]

# Title patterns that indicate a listing/search/category page (not a single product)
_NON_PRODUCT_TITLE_PATTERNS = [
    "search",
    "results",
    "collection",
    "category",
    "shop all",
    "all products",
    "marketplace",
    "best ",          # "Best coffee machines"
    "top ",           # "Top 10 espresso makers"
    "compare",
    "reviews",
    "buying guide",
]


def is_product_page(url: str) -> bool:
    """
    Returns True if the URL passes the blacklist check (no search/category patterns).
    """
    url_lower = url.lower()
    for pattern in _NON_PRODUCT_PATTERNS:
        if pattern.lower() in url_lower:
            return False
    return True


def is_single_product_url(url: str) -> bool:
    """
    Whitelist check: for known e-commerce sites, the URL must match that site's
    specific single-product page pattern. Rejects category/filtered listing pages
    that slip past the blacklist.

    Known signatures:
      Amazon  → must contain /dp/           e.g. /dp/B07N6MYL53
      Jumia   → must match  -<digits>.html  e.g. /breville-machine-12345678.html
      Noon    → must contain /p/n           e.g. /p/N12345678

    Unknown sites: returns True (no pattern to enforce).
    """
    url_lower = url.lower()
    if "amazon." in url_lower:
        return "/dp/" in url_lower
    if "jumia." in url_lower:
        return bool(re.search(r"-\d+\.html", url_lower))
    if "noon.com" in url_lower:
        return "/p/n" in url_lower or "/pdp/" in url_lower
    return True


def is_valid_product_result(title: str, url: str) -> bool:
    """
    Returns True only when ALL four checks pass:
    0. Domain:    Not a blocked domain (social media / comparison / blog).
    1. Whitelist: URL matches the known-site product page pattern.
    2. Blacklist: URL contains no search/category/listing patterns.
    3. Title:     Title does not look like a listing/collection page.
    """
    # Check 0: block social media, comparison sites, blogs
    try:
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        clean_host = hostname.removeprefix("www.")
        if any(clean_host == d or clean_host.endswith("." + d) for d in _BLOCKED_DOMAINS):
            return False
    except Exception:
        pass

    if not is_single_product_url(url):
        return False
    if not is_product_page(url):
        return False
    title_lower = title.lower()
    for pattern in _NON_PRODUCT_TITLE_PATTERNS:
        if pattern in title_lower:
            return False
    return True


def run_search_engine(
    search_client: TavilyClient,
    queries: List[str],
    score_th: float = Config.SCORE_THRESHOLD,
    max_results: int = Config.MAX_RESULTS,
) -> AllSearchResults:
    """
    Agent 2: Pure Python search engine - NO LLM required!

    Why no LLM?
    - Task is 100% mechanical: loop queries -> call Tavily -> filter by score
    - LLM adds 4000-6000 tokens overhead for ZERO benefit here
    - Python is deterministic: every query is guaranteed to execute
    - LLM can randomly skip queries or hallucinate results
    - Result: ZERO rate limit issues, faster, cheaper, more reliable
    """
    print(f"\n  Searching {len(queries)} queries (score >= {score_th})...")

    # Price-hint patterns in Tavily snippets — presence boosts the result's confidence score
    _PRICE_HINTS = re.compile(r"(EGP|\bLE\b|جنيه|ج\.م|price|\bEGP\s*[\d,]+)", re.IGNORECASE)

    # Product-page URL signatures per site give a score bonus
    def _product_score_bonus(url: str, title: str, snippet: str) -> float:
        bonus = 0.0
        u = url.lower()
        t = title.lower()
        # Strong signal: URL is definitely a product page on a known site
        if "/dp/" in u:                          bonus += 0.20  # Amazon product
        if re.search(r"-\d+\.html", u):          bonus += 0.20  # Jumia product
        if "/p/n" in u or "/pdp/" in u:          bonus += 0.20  # Noon product
        # Moderate signal: title looks like a specific product
        if any(w in t for w in ("machine", "maker", "espresso", "grinder", "nespresso")):
            bonus += 0.10
        # Moderate signal: snippet mentions a price in EGP
        if _PRICE_HINTS.search(snippet):
            bonus += 0.10
        return bonus

    all_results: List[SingleSearchResult] = []

    # Deduplicate by URL across all queries
    seen_urls: set = set()

    for query in queries:
        try:
            raw = search_client.search(query, max_results=max_results)

            query_results = []
            for item in raw.get("results", []):
                base_score = round(item.get("score", 0.0), 4)
                url   = item.get("url", "")
                title = item.get("title", "")
                snippet = item.get("content", "") or item.get("snippet", "")

                if url in seen_urls:
                    continue

                bonus = _product_score_bonus(url, title, snippet)
                final_score = round(base_score + bonus, 4)

                if final_score >= score_th and is_valid_product_result(title, url):
                    seen_urls.add(url)
                    query_results.append(SingleSearchResult(
                        title=title,
                        url=url,
                        score=final_score,
                        search_query=query,
                    ))

            all_results.extend(query_results)
            print(f"  \u2192 '{query[:60]}' \u2192 {len(query_results)} kept (of {len(raw.get('results', []))})")  

        except Exception as e:
            print(f"  \u274c Error on '{query[:50]}': {e}")

    print(f"\n  \u2713 Total product-page results: {len(all_results)}")
    return AllSearchResults(results=all_results)


# ============================================================
# AGENT 3: WEB SCRAPER (Pure Python - NO LLM overhead!)
# ============================================================

def run_web_scraper(
    scrape_client: ScrapGraphClient,
    urls: List[str],
    top_n: int = Config.TOP_SCRAPE,
) -> AllExtractedProducts:
    """
    Agent 3: Pure Python web scraper - NO CrewAI / NO Groq LLM!

    Why no LLM agent here?
    - ScrapGraph's smartscraper already uses its own internal AI to extract data
    - Wrapping it in a CrewAI Agent adds 4000+ Groq tokens of overhead per URL
    - Python loop guarantees EVERY URL is scraped (LLM agents can skip)
    - Result: ZERO Groq rate limits, reliable, faster

    ScrapGraph still uses AI - but it's THEIR AI on THEIR quota, not our Groq quota.
    """
    # Prioritise Jumia/Noon first because Amazon.eg hides prices behind JavaScript/login.
    # Jumia and Noon render prices in plain HTML, which makes extraction reliable.
    def _site_priority(u: str) -> int:
        if "jumia" in u or "noon" in u:
            return 0   # highest priority
        if "amazon" in u:
            return 1   # scrape last
        return 0

    urls_sorted = sorted(urls, key=_site_priority)
    urls_to_scrape = urls_sorted[:top_n]
    print(f"\n  Scraping {len(urls_to_scrape)} URLs (top {top_n} of {len(urls)} results)...")

    user_prompt = (
        "You are a precise data extractor for e-commerce product pages. "
        "Extract ONLY what is literally visible or encoded on the page — never invent or estimate values. "
        "\n\n"
        "PRICE EXTRACTION — most critical field. Search the entire page using ALL methods below: "
        "(1) CSS price elements: .price, .product-price, span[class*='price'], [data-price], "
        ".prc, .cost, .selling-price, .special-price, .amount. "
        "(2) JSON-LD: <script type='application/ld+json'> look for 'price' or 'offers.price'. "
        "(3) Meta tags: meta[property='product:price:amount'], meta[name='twitter:data1']. "
        "(4) Any number directly next to currency symbols: 'EGP', 'LE', 'جنيه', 'ج.م'. "
        "(5) Hidden data attributes or aria-label values containing a price number. "
        "Return product_current_price as a plain float (e.g. 15999.0). "
        "If you find NO price after all methods, return null. NEVER return 0. "
        "\n\n"
        "product_original_price: the crossed-out / was-price, or null if absent. "
        "product_discount_percentage: the % off shown (e.g. 20 for '20% off'), or null if absent. "
        "product_image_url: the main product image src URL, or null. "
        "product_specs: 1-5 key technical specs (brand, model, capacity, power, etc.) as name/value pairs."
    )
    schema = SingleExtractedProduct.model_json_schema()

    # mode='js' uses a real Playwright browser to fully render JavaScript.
    # stealth=True bypasses anti-bot fingerprinting (needed for Amazon).
    # wait=5000 gives JS price widgets 5 s to load after the page renders.
    # country='eg' routes through an Egyptian IP so sites show EGP prices.
    fetch_cfg = FetchConfig(mode="js", stealth=True, wait=5000, country="eg")

    all_products: List[SingleExtractedProduct] = []

    for i, url in enumerate(urls_to_scrape, 1):
        print(f"  [{i}/{len(urls_to_scrape)}] {url[:80]}")
        try:
            api_result = scrape_client.extract(
                prompt=user_prompt,
                url=url,
                schema=schema,
                fetch_config=fetch_cfg,
            )

            # ApiResult.data is ExtractResponse; json_data holds the extracted dict
            if api_result.status != "success" or api_result.data is None:
                print(f"       \u274c API error: {api_result.error}")
                continue

            result = api_result.data.json_data  # ExtractResponse.json_data = the dict

            if isinstance(result, dict):
                # page_url is always set from our known input URL
                result["page_url"] = url

                # Treat price=0 as not-found — 0 is never a real product price
                if result.get("product_current_price") in (0, 0.0):
                    result["product_current_price"] = None

                product = SingleExtractedProduct.model_validate(result)
                all_products.append(product)
                price_display = f"EGP {product.product_current_price:,.0f}" if product.product_current_price else "price not found"
                print(f"       \u2713 {product.product_title[:55]} | {price_display}")
            else:
                print(f"       \u274c Unexpected response type: {type(result)} | {str(result)[:100]}")

        except Exception as e:
            print(f"       \u274c Error: {str(e)[:100]}")

    print(f"\n  \u2713 Successfully scraped {len(all_products)} / {len(urls_to_scrape)} products")
    return AllExtractedProducts(products=all_products)


# ============================================================
# DATA QUALITY LAYER  (Validation → Deduplication → Ranking)
# Pure Python, no API calls, runs between Agent 3 and Agent 4.
# ============================================================

_MIN_PRICE_EGP = 500.0       # Below this → clearly wrong extraction
_MAX_PRICE_EGP = 300_000.0   # Above this → clearly wrong extraction


def _validate_products(products: AllExtractedProducts) -> AllExtractedProducts:
    """
    Rejects broken product entries:
    - Empty/missing title
    - No specs at all
    - Price outside realistic EGP range [500, 300,000]
    """
    valid = []
    for p in products.products:
        if not (p.product_title or "").strip():
            print(f"    [VALIDATE] Reject \u2014 no title: {p.page_url[:60]}")
            continue
        if not p.product_specs:
            print(f"    [VALIDATE] Reject \u2014 no specs: {p.product_title[:50]}")
            continue
        price = p.product_current_price
        if price is not None:
            if price < _MIN_PRICE_EGP:
                print(f"    [VALIDATE] Reject \u2014 price too low ({price:.0f} EGP): {p.product_title[:50]}")
                continue
            if price > _MAX_PRICE_EGP:
                print(f"    [VALIDATE] Reject \u2014 price too high ({price:,.0f} EGP): {p.product_title[:50]}")
                continue
        valid.append(p)
    removed = len(products.products) - len(valid)
    print(f"  [VALIDATE] {len(valid)} valid / {removed} rejected (of {len(products.products)})")
    return AllExtractedProducts(products=valid)


def _deduplicate_products(products: AllExtractedProducts) -> AllExtractedProducts:
    """
    Fuzzy deduplication using difflib.SequenceMatcher (stdlib, no install needed).
    If two product titles are >= 80% similar, keep the 'better' one:
      1. Prefer has-price over no-price
      2. Prefer more specs
      3. Prefer lower price (better deal)
    """
    from difflib import SequenceMatcher

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _quality(p: SingleExtractedProduct) -> tuple:
        has_price = 1 if p.product_current_price else 0
        n_specs   = len(p.product_specs)
        price_neg = -(p.product_current_price or float("inf"))  # lower price → higher priority
        return (has_price, n_specs, price_neg)

    kept: list = []
    for candidate in products.products:
        merged = False
        for idx, existing in enumerate(kept):
            if _sim(candidate.product_title, existing.product_title) >= 0.80:
                if _quality(candidate) > _quality(existing):
                    kept[idx] = candidate
                merged = True
                break
        if not merged:
            kept.append(candidate)

    removed = len(products.products) - len(kept)
    if removed:
        print(f"  [DEDUP] Removed {removed} near-duplicate(s)")
    return AllExtractedProducts(products=kept)


def _rank_products(products: AllExtractedProducts):
    """
    Deterministic scoring formula:

        total_score = (spec_quality_score * 0.6) - (normalized_price * 0.4)

    spec_quality_score (0→1):
        base  = num_specs / 5
        bonus = +0.15 if product has an image URL
        capped at 1.0

    normalized_price (0→1, min-max across products with valid prices):
        products without a price → 0.5 (neutral, does not penalise)

    Tags:
        "Best Value"   → highest total_score (best combo of quality + price)
        "Best Budget"  → lowest valid price
        "Best Premium" → highest spec_quality_score (richest specs)

    Returns (AllExtractedProducts sorted best-first, ranked_meta: list[dict])
    """
    prods = products.products
    if not prods:
        return products, []

    # spec quality
    spec_scores = []
    for p in prods:
        base  = len(p.product_specs) / 5.0
        bonus = 0.15 if p.product_image_url else 0.0
        spec_scores.append(min(base + bonus, 1.0))

    # price normalisation
    prices  = [p.product_current_price for p in prods if p.product_current_price]
    p_min   = min(prices) if prices else 1.0
    p_max   = max(prices) if prices else 1.0
    p_range = (p_max - p_min) if p_max != p_min else 1.0

    norm_prices = []
    for p in prods:
        if p.product_current_price:
            norm_prices.append((p.product_current_price - p_min) / p_range)
        else:
            norm_prices.append(0.5)

    # total score
    total_scores = [
        round(spec_scores[i] * 0.6 - norm_prices[i] * 0.4, 4)
        for i in range(len(prods))
    ]

    # tag assignment
    best_value_idx   = total_scores.index(max(total_scores))
    priced_idx       = [i for i, p in enumerate(prods) if p.product_current_price]
    best_budget_idx  = min(priced_idx, key=lambda i: prods[i].product_current_price) if priced_idx else best_value_idx
    best_premium_idx = spec_scores.index(max(spec_scores))

    ranked_meta = []
    for i, p in enumerate(prods):
        tags = []
        if i == best_value_idx:   tags.append("Best Value")
        if i == best_budget_idx:  tags.append("Best Budget")
        if i == best_premium_idx: tags.append("Best Premium")
        ranked_meta.append({
            "title":            p.product_title,
            "spec_score":       round(spec_scores[i], 4),
            "normalized_price": round(norm_prices[i], 4),
            "total_score":      total_scores[i],
            "price_egp":        p.product_current_price,
            "tags":             tags,
        })

    # sort best-first
    sorted_pairs = sorted(zip(prods, ranked_meta, total_scores), key=lambda x: x[2], reverse=True)
    sorted_prods = [t[0] for t in sorted_pairs]
    sorted_meta  = [t[1] for t in sorted_pairs]

    print(f"  [RANK] Best: {sorted_prods[0].product_title[:55]} (score={sorted_pairs[0][2]:.4f})")
    for m in sorted_meta:
        if m["tags"]:
            print(f"  [RANK] {m['tags']} \u2192 {m['title'][:50]}")

    return AllExtractedProducts(products=sorted_prods), sorted_meta


def _export_to_pdf(html_path: str) -> str:
    """
    Converts the HTML report to PDF using Playwright (Chromium).
    Playwright is already installed as a ScrapGraph dependency \u2014 no extra install needed.
    Returns the PDF path, or empty string on failure (non-fatal).
    """
    pdf_path = html_path.replace(".html", ".pdf")
    abs_html = os.path.abspath(html_path)

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            pg      = browser.new_page()
            pg.goto(f"file:///{abs_html}")
            pg.wait_for_load_state("networkidle")
            pg.pdf(
                path=pdf_path,
                format="A4",
                margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
                print_background=True,
            )
            browser.close()
        print(f"  [PDF] Exported: {pdf_path}")
        return pdf_path
    except Exception as exc:
        print(f"  [PDF] Export skipped ({str(exc)[:80]})")
        return ""


# ============================================================
# AGENT 4: PROCUREMENT REPORT AUTHOR (Pure Python + litellm - NO CrewAI overhead!)
# ============================================================

def run_procurement_report(
    products: AllExtractedProducts,
    product_name: str,
    ranked_meta: list,
) -> str:
    """
    Agent 4 — Hybrid approach:
    - LLM writes ONLY short text content (Executive Summary, Conclusion, Recommendation reasons).
      This is a tiny JSON response (~400 tokens) so no rate-limit issues and no PHP hallucination.
    - Python generates the FULL HTML with Bootstrap 5 + Chart.js.
      Data is injected directly by Python — no template engine, no PHP, always correct.
    """
    import litellm
    from urllib.parse import urlparse

    TOP_N = 7
    top_prods = products.products[:TOP_N]
    top_meta  = ranked_meta[:TOP_N]

    # ── Step 1: LLM generates ONLY short text (not HTML) ─────────────────────
    slim_for_llm = [
        {
            "title": p.product_title[:55],
            "price": p.product_current_price,
            "tags":  (top_meta[i]["tags"] if i < len(top_meta) else []),
        }
        for i, p in enumerate(top_prods)
    ]
    text_prompt = "\n".join([
        "You are a procurement analyst. Return ONLY a valid JSON object, no extra text.",
        f"Product: {product_name} (Egyptian e-commerce market)",
        "Products ranked best-first: " + json.dumps(slim_for_llm, ensure_ascii=False),
        "",
        "Return exactly this JSON (fill the string values):",
        '{"executive_summary":"2-3 sentences overview",',
        ' "conclusion":"2-3 sentences procurement recommendation",',
        ' "best_value_reason":"1 sentence",',
        ' "best_budget_reason":"1 sentence",',
        ' "best_premium_reason":"1 sentence"}',
        "No markdown. No explanation. JSON only.",
    ])

    # Safe defaults in case LLM fails
    text = {
        "executive_summary": f"This report analyzes {len(top_prods)} {product_name} products available on Egyptian e-commerce platforms.",
        "conclusion": "Based on the collected data, the ranked products above offer a range of options from budget-friendly to premium quality.",
        "best_value_reason": "Achieves the best balance of specification quality and competitive price.",
        "best_budget_reason": "Lowest price among products with confirmed EGP pricing.",
        "best_premium_reason": "Richest specification set available in this product category.",
    }
    try:
        resp = litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": text_prompt}],
            api_key=Config.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        text.update(json.loads(raw))
        print("  [Agent 4] LLM text generated.")
    except Exception as exc:
        print(f"  [Agent 4] LLM text failed ({str(exc)[:60]}), using defaults.")

    # ── Step 2: Python builds full HTML — no LLM, no PHP ─────────────────────
    priced   = [p for p in top_prods if p.product_current_price]
    unpriced = [p for p in top_prods if not p.product_current_price]

    # Table rows
    rows_html_parts = []
    for i, p in enumerate(top_prods):
        img_tag = (
            f'<img src="{p.product_image_url}" width="50" height="50" '
            f'style="object-fit:contain" onerror="this.style.display=\'none\'">'
            if p.product_image_url else '<span class="text-muted small">—</span>'
        )
        try:
            domain = urlparse(p.page_url).hostname or p.page_url
        except Exception:
            domain = p.page_url[:30]
        price_cell = f'<strong>EGP {p.product_current_price:,.0f}</strong>' if p.product_current_price else '<span class="text-muted">N/A</span>'
        orig_cell  = f'EGP {p.product_original_price:,.0f}' if p.product_original_price else '—'
        if p.product_discount_percentage:
            pct = p.product_discount_percentage
            cls = "success" if pct < 10 else ("warning" if pct < 20 else "danger")
            disc_cell = f'<span class="badge bg-{cls}">{pct:.0f}% off</span>'
        else:
            disc_cell = '—'
        specs_html = "".join(
            f'<small class="d-block">{s.specification_name}: <b>{s.specification_value}</b></small>'
            for s in p.product_specs[:3]
        )
        rows_html_parts.append(
            f'<tr>'
            f'<td class="text-center">{img_tag}</td>'
            f'<td><a href="{p.page_url}" target="_blank" class="fw-semibold text-decoration-none">{p.product_title[:75]}</a></td>'
            f'<td><small class="badge bg-secondary">{domain}</small></td>'
            f'<td>{price_cell}</td>'
            f'<td><small>{orig_cell}</small></td>'
            f'<td>{disc_cell}</td>'
            f'<td>{specs_html}</td>'
            f'<td><a href="{p.page_url}" target="_blank" class="btn btn-sm btn-outline-primary">View</a></td>'
            f'</tr>'
        )
    table_html = "\n".join(rows_html_parts)

    # Recommendation cards
    tag_reasons = {
        "Best Value":   text.get("best_value_reason", ""),
        "Best Budget":  text.get("best_budget_reason", ""),
        "Best Premium": text.get("best_premium_reason", ""),
    }
    tag_colors  = {"Best Value": "success", "Best Budget": "info", "Best Premium": "warning"}
    rec_parts   = []
    for m in top_meta:
        if not m.get("tags"):
            continue
        prod = next((p for p in top_prods if p.product_title == m["title"]), None)
        price_str = f'EGP {prod.product_current_price:,.0f}' if prod and prod.product_current_price else 'N/A'
        for tag in m["tags"]:
            color = tag_colors.get(tag, "secondary")
            rec_parts.append(
                f'<div class="col-md-4 mb-3">'
                f'<div class="card border-{color} h-100">'
                f'<div class="card-header bg-{color} text-white fw-bold">{tag}</div>'
                f'<div class="card-body">'
                f'<h6>{m["title"][:60]}</h6>'
                f'<p class="small text-muted mb-2">{tag_reasons.get(tag, "")})</p>'
                f'<span class="badge bg-light text-dark border me-1">Price: {price_str}</span>'
                f'<span class="badge bg-light text-dark border me-1">Score: {m["total_score"]:.3f}</span>'
                f'<span class="badge bg-light text-dark border">Spec: {m["spec_score"]:.2f}</span>'
                f'</div></div></div>'
            )
    rec_html = "\n".join(rec_parts) if rec_parts else '<p class="text-muted">No tagged recommendations.</p>'

    # Chart data (Python-injected — no PHP, always valid JS)
    chart_labels  = json.dumps([p.product_title[:22].replace("'", "").replace('"', '') for p in priced])
    chart_prices  = json.dumps([p.product_current_price for p in priced])
    spec_labels   = json.dumps([m["title"][:22].replace("'", "").replace('"', '') for m in top_meta])
    spec_scores   = json.dumps([m["spec_score"] for m in top_meta])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{product_name} — Procurement Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ background:#f8f9fa; }}
    .stat-card {{ border-left:4px solid; }}
    .stat-card.total   {{ border-color:#0d6efd; }}
    .stat-card.priced  {{ border-color:#198754; }}
    .stat-card.unpriced{{ border-color:#ffc107; }}
    .table thead th {{ background:#212529; color:#fff; white-space:nowrap; }}
    .chart-wrap {{ background:#fff; border-radius:8px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
    footer {{ background:#212529; color:#adb5bd; padding:14px 0; margin-top:40px; text-align:center; }}
  </style>
</head>
<body>
  <nav class="navbar navbar-dark bg-dark mb-4">
    <div class="container">
      <span class="navbar-brand fw-bold">ResearchFlow AI</span>
      <span class="navbar-text text-light">{product_name} — Procurement Report</span>
    </div>
  </nav>

  <div class="container pb-5">

    <!-- Stat cards -->
    <div class="row g-3 mb-4">
      <div class="col-md-4">
        <div class="card stat-card total h-100 shadow-sm">
          <div class="card-body"><h6 class="text-muted mb-1">Total Products</h6>
            <h2 class="mb-0 fw-bold text-primary">{len(top_prods)}</h2></div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card stat-card priced h-100 shadow-sm">
          <div class="card-body"><h6 class="text-muted mb-1">With Price</h6>
            <h2 class="mb-0 fw-bold text-success">{len(priced)}</h2></div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card stat-card unpriced h-100 shadow-sm">
          <div class="card-body"><h6 class="text-muted mb-1">Price Not Found</h6>
            <h2 class="mb-0 fw-bold text-warning">{len(unpriced)}</h2></div>
        </div>
      </div>
    </div>

    <!-- Executive Summary -->
    <div class="card shadow-sm mb-4">
      <div class="card-header fw-bold bg-dark text-white">Executive Summary</div>
      <div class="card-body"><p class="mb-0">{text['executive_summary']}</p></div>
    </div>

    <!-- Product Table -->
    <div class="card shadow-sm mb-4">
      <div class="card-header fw-bold bg-dark text-white">Product Comparison Table</div>
      <div class="card-body p-0">
        <div class="table-responsive">
          <table class="table table-striped table-hover align-middle mb-0">
            <thead>
              <tr>
                <th>Image</th><th>Product</th><th>Store</th>
                <th>Price (EGP)</th><th>Original</th><th>Discount</th>
                <th>Key Specs</th><th>Link</th>
              </tr>
            </thead>
            <tbody>
{table_html}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Charts -->
    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="chart-wrap shadow-sm">
          <h6 class="fw-bold mb-3">Price Comparison (EGP)</h6>
          <canvas id="priceChart"></canvas>
        </div>
      </div>
      <div class="col-md-6">
        <div class="chart-wrap shadow-sm">
          <h6 class="fw-bold mb-3">Spec Quality Score (0 – 1)</h6>
          <canvas id="specChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Recommendations -->
    <div class="card shadow-sm mb-4">
      <div class="card-header fw-bold bg-dark text-white">Recommendations</div>
      <div class="card-body">
        <div class="row">{rec_html}</div>
      </div>
    </div>

    <!-- Conclusion -->
    <div class="card shadow-sm mb-4">
      <div class="card-header fw-bold bg-dark text-white">Conclusion</div>
      <div class="card-body"><p class="mb-0">{text['conclusion']}</p></div>
    </div>

  </div>

  <footer><small>Generated by ResearchFlow AI &mdash; {product_name} &mdash; {len(top_prods)} products analysed</small></footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    // All data injected by Python — no PHP, always valid
    try {{
      new Chart(document.getElementById('priceChart'), {{
        type: 'bar',
        data: {{
          labels: {chart_labels},
          datasets: [{{
            label: 'Price (EGP)',
            data: {chart_prices},
            backgroundColor: 'rgba(13,110,253,0.65)',
            borderColor: 'rgba(13,110,253,1)',
            borderWidth: 1
          }}]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: false }} }} }}
      }});
    }} catch(e) {{ console.warn('Price chart:', e); }}
    try {{
      new Chart(document.getElementById('specChart'), {{
        type: 'bar',
        data: {{
          labels: {spec_labels},
          datasets: [{{
            label: 'Spec Score',
            data: {spec_scores},
            backgroundColor: 'rgba(25,135,84,0.65)',
            borderColor: 'rgba(25,135,84,1)',
            borderWidth: 1
          }}]
        }},
        options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ min: 0, max: 1 }} }} }}
      }});
    }} catch(e) {{ console.warn('Spec chart:', e); }}
  </script>
</body>
</html>"""
    return html


def main():
    """Main execution function"""

    print("=" * 60)
    print("ResearchFlow AI - Multi-Agent System")
    print("Agent 1: CrewAI + LLM  |  Agent 2: Pure Python + Tavily  |  Agent 3: Pure Python + ScrapGraph")
    print("=" * 60)
    print()

    validate_config()
    initialize_environment()

    # Get product to research — from CLI argument or interactive prompt
    if len(sys.argv) > 1:
        product_name = " ".join(sys.argv[1:])
    else:
        product_name = input("\n  Enter the product to research (e.g. 'coffee machine for the office'): ").strip()
        if not product_name:
            product_name = "coffee machine for the office"
    print(f"\n  Researching: {product_name}")

    llm_agent1, search_client, scrape_client = initialize_clients()

    # ============================================================
    # STEP 1: Agent 1 - Generate search queries (LLM needed)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: Agent 1 - Generating search queries (LLM)...")
    print("=" * 60)

    search_queries_agent = create_search_queries_agent(llm_agent1)
    search_queries_task = create_search_queries_task(search_queries_agent)

    crew_agent1 = Crew(
        agents=[search_queries_agent],
        tasks=[search_queries_task],
        process=Process.sequential,
    )

    crew_agent1.kickoff(inputs={
        "product_name": product_name,
        # Jumia and Noon listed first: they display prices in plain HTML without login.
        # Amazon.eg is last: prices are often hidden behind JavaScript / sign-in.
        "websites_list": ["www.jumia.com.eg", "www.noon.com/egypt-en", "www.amazon.eg"],
        "country_name": "Egypt",
        "no_keywords": Config.NO_KEYWORDS,
        "language": "English",
    })

    # Read Agent 1 output from file
    queries_file = os.path.join(Config.OUTPUT_DIR, "step_1_suggested_search_queries.json")
    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    search_queries = queries_data.get("queries", [])

    print(f"\n\u2713 Agent 1 produced {len(search_queries)} queries:")
    for i, q in enumerate(search_queries, 1):
        print(f"  {i}. {q}")

    # ============================================================
    # STEP 2: Agent 2 - Web search (Pure Python, NO LLM, NO rate limits!)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: Agent 2 - Searching the web (Pure Python, NO LLM)...")
    print("=" * 60)

    search_results = run_search_engine(
        search_client=search_client,
        queries=search_queries,
        score_th=Config.SCORE_THRESHOLD,
        max_results=Config.MAX_RESULTS,
    )

    # Save Agent 2 output to JSON
    output_file = os.path.join(Config.OUTPUT_DIR, "step_2_search_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(search_results.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"\n\u2713 Agent 2 saved {len(search_results.results)} results to {output_file}")

    # ============================================================
    # STEP 3: Agent 3 - Scrape product pages (Pure Python + ScrapGraph)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: Agent 3 - Scraping product pages (Pure Python, ScrapGraph)...")
    print("=" * 60)

    # Extract URLs from Agent 2 results
    all_urls = [r.url for r in search_results.results]
    print(f"  Found {len(all_urls)} URLs from Agent 2")

    extracted_products = run_web_scraper(
        scrape_client=scrape_client,
        urls=all_urls,
        top_n=Config.TOP_SCRAPE,
    )

    # Save Agent 3 output to JSON
    scrape_output_file = os.path.join(Config.OUTPUT_DIR, "step_3_extracted_products.json")
    with open(scrape_output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_products.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"\n\u2713 Agent 3 saved {len(extracted_products.products)} products to {scrape_output_file}")

    # ============================================================
    # STEP 4: Agent 4 - Generate procurement report (Pure Python + litellm, NO CrewAI!)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: Agent 4 - Generating procurement report (Direct litellm call, NO CrewAI)...")
    print("=" * 60)

    if len(extracted_products.products) == 0:
        print("  \u26a0 No products scraped \u2014 skipping report generation.")
    else:
        # ── Data Quality Pipeline ──────────────────────────────────────────
        print("\n  Running data quality pipeline (validate \u2192 dedup \u2192 rank)...")
        validated_products = _validate_products(extracted_products)

        if not validated_products.products:
            print("  \u26a0 No products passed validation \u2014 skipping report.")
        else:
            deduplicated_products        = _deduplicate_products(validated_products)
            ranked_products, ranked_meta = _rank_products(deduplicated_products)

            # Save ranked data for transparency / debugging
            ranked_file = os.path.join(Config.OUTPUT_DIR, "step_3b_ranked_products.json")
            with open(ranked_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "products": ranked_products.model_dump()["products"],
                        "ranking_metadata": ranked_meta,
                    },
                    f, indent=2, ensure_ascii=False,
                )
            print(f"  \u2713 Ranked data saved to {ranked_file}")

            # ── Generate HTML report ───────────────────────────────────────
            html_report = run_procurement_report(ranked_products, product_name, ranked_meta)
            report_path = os.path.join(Config.OUTPUT_DIR, "step_4_procurement_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_report)
            print(f"\n\u2713 Agent 4 saved procurement report to {report_path}")

            # ── Export to PDF (non-fatal if Playwright not configured) ─────
            _export_to_pdf(report_path)

    agentops.end_session("Success")

    print("\n" + "=" * 60)
    print("\u2713 All agents completed successfully!")
    print(f"\u2713 Agent 1:      {Config.OUTPUT_DIR}/step_1_suggested_search_queries.json")
    print(f"\u2713 Agent 2:      {Config.OUTPUT_DIR}/step_2_search_results.json")
    print(f"\u2713 Agent 3:      {Config.OUTPUT_DIR}/step_3_extracted_products.json")
    print(f"\u2713 Data Quality: {Config.OUTPUT_DIR}/step_3b_ranked_products.json")
    print(f"\u2713 Agent 4 HTML: {Config.OUTPUT_DIR}/step_4_procurement_report.html")
    print(f"\u2713 Agent 4 PDF:  {Config.OUTPUT_DIR}/step_4_procurement_report.pdf")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        agentops.end_session("Fail")
        raise
