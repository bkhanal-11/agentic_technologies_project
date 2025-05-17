import asyncio
import json
import uuid
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from utils.logger import logger

async def send_bdi_message(from_id, to_id, literal_type, content, password="password"):
    """
    Send a message between BDI agents using a temporary agent
    
    Args:
        from_id: Sender agent JID
        to_id: Receiver agent JID
        literal_type: Type of the literal (e.g., 'search_params')
        content: Content to send - will be properly formatted as AgentSpeak literal
        password: Password for the temporary agent
    """
    # Create a unique ID for the sender
    sender_id = f"{from_id.split('@')[0]}_sender_{uuid.uuid4().hex[:8]}@{from_id.split('@')[1]}"
    
    # Format content properly for ASL
    if isinstance(content, str):
        # String content needs quotes in AgentSpeak, and escape any existing quotes
        content_safe = content.replace("'", "\\'")
        message_body = f'{literal_type}(\'{content_safe}\')'
    elif isinstance(content, dict) or isinstance(content, list):
        # Convert dictionaries or lists to JSON strings, then quote
        json_str = json.dumps(content)
        message_body = f'{literal_type}(\'{json_str}\')'
    else:
        # For other types, convert to string and quote
        content_str = str(content)
        message_body = f'{literal_type}(\'{content_str}\')'
    
    logger.debug(f"Formatted message: {message_body}")
    
    class MessageSender(Agent):
        def __init__(self, jid, password, to_jid, message_body):
            super().__init__(jid, password)
            self.to_jid = to_jid
            self.message_body = message_body
        
        class SendMessage(OneShotBehaviour):
            async def run(self):
                msg = Message(
                    to=self.agent.to_jid,
                    body=self.agent.message_body,
                    metadata={
                        "performative": "BDI",
                        "ilf_type": "tell"
                    }
                )
                
                await self.send(msg)
                logger.info(f"Message sent from {self.agent.jid} to {self.agent.to_jid}: {self.agent.message_body[:100]}...")
                await asyncio.sleep(1)  # Give time for message to be processed
                await self.agent.stop()
        
        async def setup(self):
            self.add_behaviour(self.SendMessage())
    
    # Create and start the sender agent
    sender = MessageSender(sender_id, password, to_id, message_body)
    await sender.start()