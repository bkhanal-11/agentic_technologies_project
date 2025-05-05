from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

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
        try:
            root = ET.fromstring(xml_response)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}  # Define the namespace

            results = []
            for entry in root.findall('atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
                
                summary_elem = entry.find('atom:summary', ns)
                summary = summary_elem.text.strip() if summary_elem is not None else ""
                
                # Extract author information
                authors = []
                for author_elem in entry.findall('atom:author', ns):
                    name_elem = author_elem.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                
                # Extract published date
                published_elem = entry.find('atom:published', ns)
                published = published_elem.text.strip() if published_elem is not None else ""
                
                # Extract links
                pdf_link = ""
                page_link = ""
                for link in entry.findall('atom:link', ns):
                    href = link.get('href')
                    rel = link.get('rel')
                    title_attr = link.get('title')
                    
                    if rel == "alternate":
                        page_link = href
                    elif title_attr == "pdf":
                        pdf_link = href
                
                # Extract categories/tags
                categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
                
                # Extract paper ID
                id_elem = entry.find('atom:id', ns)
                arxiv_id = id_elem.text.split("/")[-1] if id_elem is not None else ""
                
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