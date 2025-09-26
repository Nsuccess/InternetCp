import os
import json
import sys
from dotenv import load_dotenv
from fastmcp import FastMCP
from firecrawl import FirecrawlApp
from tavily import TavilyClient
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the MCP server
server = FastMCP(
    name="Universal Search Assistant"
)

def search_with_tavily(query: str, max_results: int = 5):
    """Fallback search using Tavily"""
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        search_result = tavily_client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_images=False
        )
        
        results = []
        for result in search_result.get('results', []):
            results.append({
                'title': result.get('title', 'No title'),
                'url': result.get('url', 'No URL'),
                'content': result.get('content', 'No content')
            })
        
        return results
    except Exception as e:
        print(f"🛠️ DEBUG: Tavily search failed: {e}", file=sys.stderr)
        return []

@server.tool(
    name="search_web",
    description="Performs a web search using multiple providers and returns a concise, well-formatted answer using GPT-4o-mini"
)
def search_web(query: str, max_results: int = 5):
    """Search the web using FireCrawl with fallbacks and process results with GPT-4o-mini"""
    print(f"🛠️ DEBUG: search_web called with query: {query}", file=sys.stderr)
    sys.stderr.flush()
    
    # Initialize clients
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []
    search_method_used = "Unknown"
    
    # Try FireCrawl first
    try:
        firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        search_result = firecrawl_app.search(query)
        
        if hasattr(search_result, 'data') and search_result.data:
            firecrawl_results = search_result.data
            if isinstance(firecrawl_results, list) and len(firecrawl_results) > 0:
                for result in firecrawl_results:
                    title = getattr(result, 'title', None) or 'No title'
                    url = getattr(result, 'url', None) or 'No URL'
                    content = getattr(result, 'markdown', None) or getattr(result, 'content', None) or 'No content'
                    
                    # Only use if we have actual content
                    if title != 'No title' or url != 'No URL' or (content and content != 'No content' and len(content) > 10):
                        results.append({
                            'title': title,
                            'url': url,
                            'content': content
                        })
                
                if results:
                    search_method_used = "FireCrawl"
                    print(f"🛠️ DEBUG: FireCrawl returned {len(results)} useful results", file=sys.stderr)
    except Exception as e:
        print(f"🛠️ DEBUG: FireCrawl failed: {e}", file=sys.stderr)
    
    # If FireCrawl didn't work or returned poor results, try Tavily
    if not results:
        print("🛠️ DEBUG: Trying Tavily fallback", file=sys.stderr)
        results = search_with_tavily(query, max_results)
        if results:
            search_method_used = "Tavily"
            print(f"🛠️ DEBUG: Tavily returned {len(results)} results", file=sys.stderr)
    
    # If we still have no results, return an informative message
    if not results:
        return f"""I was unable to find search results for "{query}" using available search providers (FireCrawl, Tavily). This could be due to:
1. API limitations or issues
2. The query being too specific or recent
3. Network connectivity issues

Please try:
- Rephrasing your query
- Checking your internet connection
- Verifying your API keys are configured correctly
- Trying again in a few minutes

For ManCity match information specifically, I recommend checking:
- Official Manchester City FC website
- BBC Sport
- ESPN
- Sky Sports"""
    
    # Format search results for the LLM
    context = f"Search results (via {search_method_used}):\n"
    
    for i, result in enumerate(results[:max_results], 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        content = result.get('content', 'No content')
        
        # Truncate very long content
        if len(str(content)) > 1500:
            content = str(content)[:1500] + "..."
            
        context += f"[{i}] Source: {url}\n"
        context += f" Title: {title}\n"
        context += f" Content: {content}\n\n"
    
    # Add citation information
    sources = f"\n\nSources (via {search_method_used}):\n"
    for i, result in enumerate(results[:max_results], 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        sources += f"[{i}] {title} - {url}\n"
    
    # Create prompt for GPT-4o-mini
    prompt = f"""You are a helpful research assistant that provides accurate, concise, and well-sourced answers.
Using the search results below, provide a comprehensive answer to the query: "{query}"

{context}

Instructions:
1. Provide a clear, well-structured answer
2. Cite sources using the numbered references above
3. If the search results don't contain relevant information, state this clearly
4. Keep the answer focused and avoid speculation
5. Note that results were obtained via {search_method_used}

Answer:"""
    
    # Get response from GPT-4o-mini
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research assistant that provides accurate information based on search results."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    
    # Return the answer with sources
    result = response.choices[0].message.content.strip() + sources
    print(f"🛠️ DEBUG: Generated result using {search_method_used} (first 100 chars): {result[:100]}...", file=sys.stderr)
    sys.stderr.flush()
    
    return result

if __name__ == "__main__":
    print("🔧 DEBUG: Starting FastMCP server with STDIO transport...", file=sys.stderr)
    sys.stderr.flush()
    server.run(transport="stdio")