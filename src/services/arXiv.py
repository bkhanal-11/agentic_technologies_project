from typing import List, Dict, Any, Optional
import aiohttp

from utils.logger import logger

class ArxivService:
    """Service to search and retrieve papers from arXiv"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search for papers on arXiv"""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    # arXiv API returns XML, we need to parse it
                    xml_response = await response.text()
                    return self._parse_arxiv_response(xml_response)
                else:
                    logger.error(f"arXiv API error: {await response.text()}")
                    return []
    
    def _parse_arxiv_response(self, xml_response: str) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response into structured data"""
        # Using minidom to parse XML
        from xml.dom import minidom
        
        try:
            dom = minidom.parseString(xml_response)
            entries = dom.getElementsByTagName("entry")
            
            results = []
            for entry in entries:
                # Extract paper details
                title_elem = entry.getElementsByTagName("title")
                title = title_elem[0].firstChild.nodeValue.strip() if title_elem else "Unknown Title"
                
                summary_elem = entry.getElementsByTagName("summary")
                summary = summary_elem[0].firstChild.nodeValue.strip() if summary_elem else ""
                
                # Extract author information
                authors = []
                author_elems = entry.getElementsByTagName("author")
                for author_elem in author_elems:
                    name_elem = author_elem.getElementsByTagName("name")
                    if name_elem:
                        authors.append(name_elem[0].firstChild.nodeValue.strip())
                
                # Extract published date
                published_elem = entry.getElementsByTagName("published")
                published = published_elem[0].firstChild.nodeValue.strip() if published_elem else ""
                
                # Extract link to paper
                links = entry.getElementsByTagName("link")
                pdf_link = ""
                page_link = ""
                for link in links:
                    href = link.getAttribute("href")
                    rel = link.getAttribute("rel")
                    title = link.getAttribute("title")
                    
                    if rel == "alternate":
                        page_link = href
                    elif title == "pdf":
                        pdf_link = href
                
                # Extract categories/tags
                categories = []
                category_elems = entry.getElementsByTagName("category")
                for cat_elem in category_elems:
                    categories.append(cat_elem.getAttribute("term"))
                
                # Extract paper ID
                id_elem = entry.getElementsByTagName("id")
                arxiv_id = ""
                if id_elem:
                    id_url = id_elem[0].firstChild.nodeValue.strip()
                    # Extract ID from URL (e.g., http://arxiv.org/abs/2104.08696v1 -> 2104.08696v1)
                    arxiv_id = id_url.split("/")[-1]
                
                paper = {
                    "id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "authors": authors,
                    "published": published,
                    "pdf_url": pdf_link,
                    "page_url": page_link,
                    "categories": categories
                }
                
                results.append(paper)
            
            return results
        
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {str(e)}")
            return []