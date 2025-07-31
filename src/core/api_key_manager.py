"""
API Key Management System for Congressional Trading Intelligence
Secure handling of API keys for multiple services
"""

import os
import json
import base64
from typing import Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIKeyConfig:
    """Configuration for API key management"""
    config_file: str = "api_keys.json"
    env_prefix: str = "TRADING_"
    encrypt_keys: bool = False
    
class APIKeyManager:
    """Secure API key management for trading system"""
    
    def __init__(self, config: APIKeyConfig = None):
        self.config = config or APIKeyConfig()
        self.keys: Dict[str, str] = {}
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from environment variables and config file"""
        # Load from environment variables first
        self._load_from_env()
        
        # Load from config file (if exists)
        self._load_from_file()
        
        logger.info(f"Loaded {len(self.keys)} API keys")
    
    def _load_from_env(self):
        """Load API keys from environment variables"""
        env_keys = {
            'FINNHUB_API_KEY': 'finnhub',
            'FMP_API_KEY': 'financial_modeling_prep',
            'ALPHA_VANTAGE_API_KEY': 'alpha_vantage',
            'ALPACA_API_KEY': 'alpaca_key',
            'ALPACA_SECRET_KEY': 'alpaca_secret',
            'QUIVER_API_KEY': 'quiver',
            'CONGRESS_API_KEY': 'congress_api',
            'TWITTER_API_KEY': 'twitter',
            'REDDIT_API_KEY': 'reddit',
            'NEWS_API_KEY': 'news_api'
        }
        
        for env_var, key_name in env_keys.items():
            value = os.getenv(env_var) or os.getenv(f"{self.config.env_prefix}{env_var}")
            if value:
                self.keys[key_name] = value
                logger.debug(f"Loaded {key_name} from environment")
    
    def _load_from_file(self):
        """Load API keys from configuration file"""
        config_path = Path(self.config.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_keys = json.load(f)
                
                # Decrypt if needed
                if self.config.encrypt_keys:
                    file_keys = self._decrypt_keys(file_keys)
                
                self.keys.update(file_keys)
                logger.info(f"Loaded keys from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
    
    def save_keys(self, keys: Dict[str, str]):
        """Save API keys to configuration file"""
        try:
            # Merge with existing keys
            self.keys.update(keys)
            
            # Encrypt if needed
            save_keys = self._encrypt_keys(self.keys) if self.config.encrypt_keys else self.keys
            
            with open(self.config.config_file, 'w') as f:
                json.dump(save_keys, f, indent=2)
            
            logger.info(f"Saved {len(keys)} API keys to {self.config.config_file}")
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def get_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        return self.keys.get(service)
    
    def has_key(self, service: str) -> bool:
        """Check if API key exists for service"""
        return service in self.keys and bool(self.keys[service])
    
    def get_all_keys(self) -> Dict[str, str]:
        """Get all API keys (for internal use only)"""
        return self.keys.copy()
    
    def validate_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        required_keys = [
            'finnhub',
            'financial_modeling_prep', 
            'alpha_vantage',
            'alpaca_key',
            'alpaca_secret'
        ]
        
        validation_results = {}
        for key in required_keys:
            validation_results[key] = self.has_key(key)
        
        return validation_results
    
    def get_missing_keys(self) -> list:
        """Get list of missing required API keys"""
        validation = self.validate_keys()
        return [key for key, valid in validation.items() if not valid]
    
    def _encrypt_keys(self, keys: Dict[str, str]) -> Dict[str, str]:
        """Simple base64 encoding (not secure encryption)"""
        encrypted = {}
        for key, value in keys.items():
            encrypted[key] = base64.b64encode(value.encode()).decode()
        return encrypted
    
    def _decrypt_keys(self, keys: Dict[str, str]) -> Dict[str, str]:
        """Simple base64 decoding"""
        decrypted = {}
        for key, value in keys.items():
            try:
                decrypted[key] = base64.b64decode(value.encode()).decode()
            except:
                decrypted[key] = value  # Fallback for unencrypted values
        return decrypted
    
    def create_api_configs(self) -> Dict[str, Any]:
        """Create API configuration objects for different services"""
        configs = {}
        
        # Finnhub configuration
        if self.has_key('finnhub'):
            configs['finnhub'] = {
                'api_key': self.get_key('finnhub'),
                'base_url': 'https://finnhub.io/api/v1',
                'rate_limit': 60  # requests per minute
            }
        
        # Financial Modeling Prep
        if self.has_key('financial_modeling_prep'):
            configs['fmp'] = {
                'api_key': self.get_key('financial_modeling_prep'),
                'base_url': 'https://financialmodelingprep.com/api',
                'rate_limit': 250
            }
        
        # Alpha Vantage
        if self.has_key('alpha_vantage'):
            configs['alpha_vantage'] = {
                'api_key': self.get_key('alpha_vantage'),
                'base_url': 'https://www.alphavantage.co/query',
                'rate_limit': 5
            }
        
        # Alpaca Trading
        if self.has_key('alpaca_key') and self.has_key('alpaca_secret'):
            configs['alpaca'] = {
                'api_key': self.get_key('alpaca_key'),
                'secret_key': self.get_key('alpaca_secret'),
                'base_url': 'https://paper-api.alpaca.markets',  # Paper trading by default
                'data_url': 'https://data.alpaca.markets'
            }
        
        return configs
    
    def print_status(self):
        """Print API key status for debugging"""
        print("=== API KEY STATUS ===")
        validation = self.validate_keys()
        
        for service, valid in validation.items():
            status = "✅ READY" if valid else "❌ MISSING"
            print(f"{service.upper()}: {status}")
        
        missing = self.get_missing_keys()
        if missing:
            print(f"\nMissing keys: {', '.join(missing)}")
            print("Set environment variables or use save_keys() method")
        else:
            print("\n🎉 All required API keys are configured!")

def setup_api_keys_interactive():
    """Interactive setup for API keys"""
    manager = APIKeyManager()
    
    print("=== Congressional Trading System API Key Setup ===")
    print("Enter your API keys (press Enter to skip):\n")
    
    api_services = {
        'finnhub': 'Finnhub (Congressional trading data)',
        'financial_modeling_prep': 'Financial Modeling Prep (Senate trading)',
        'alpha_vantage': 'Alpha Vantage (Market data)',
        'alpaca_key': 'Alpaca API Key (Trading)',
        'alpaca_secret': 'Alpaca Secret Key (Trading)',
        'quiver': 'Quiver Quantitative (Congressional data)',
        'congress_api': 'Congress.gov API (Legislative data)',
        'twitter': 'Twitter API (Social sentiment)',
        'reddit': 'Reddit API (Social sentiment)',
        'news_api': 'News API (News sentiment)'
    }
    
    new_keys = {}
    for key, description in api_services.items():
        if not manager.has_key(key):
            value = input(f"{description}: ").strip()
            if value:
                new_keys[key] = value
    
    if new_keys:
        manager.save_keys(new_keys)
        print(f"\n✅ Saved {len(new_keys)} API keys")
    
    manager.print_status()
    return manager

# Example usage and testing
if __name__ == "__main__":
    # Test the API key manager
    manager = APIKeyManager()
    manager.print_status()
    
    # Example of setting keys programmatically
    example_keys = {
        'finnhub': 'your_finnhub_key_here',
        'financial_modeling_prep': 'your_fmp_key_here',
        'alpha_vantage': 'your_av_key_here',
        'alpaca_key': 'your_alpaca_key_here',
        'alpaca_secret': 'your_alpaca_secret_here'
    }
    
    print("\nExample: manager.save_keys(example_keys)")
    print("Or run: python api_key_manager.py --interactive")

