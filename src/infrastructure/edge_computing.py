"""
APEX Edge Computing Architecture
Distributed Congressional Trading Intelligence Processing
"""

import asyncio
import aiohttp
import json
import numpy as np
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import websockets
import logging

@dataclass
class EdgeNode:
    """Edge computing node specification"""
    node_id: str
    region: str
    capabilities: List[str]
    max_concurrent_tasks: int
    current_load: float
    latency_ms: float
    available: bool
    last_heartbeat: datetime

class EdgeComputingOrchestrator:
    """
    Central orchestrator for distributed edge computing
    Manages task distribution and load balancing
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.active_tasks: Dict[str, Any] = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Performance metrics
        self.metrics = {
            'total_tasks_processed': 0,
            'average_processing_time': 0.0,
            'current_system_load': 0.0,
            'node_efficiency_scores': {}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start_orchestrator(self):
        """Start the edge computing orchestrator"""
        self.logger.info("🌐 Starting APEX Edge Computing Orchestrator")
        
        # Initialize edge nodes
        await self.initialize_edge_nodes()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.task_distribution_loop()),
            asyncio.create_task(self.node_health_monitoring()),
            asyncio.create_task(self.performance_monitoring())
        ]
        
        await asyncio.gather(*tasks)
    
    async def initialize_edge_nodes(self):
        """Initialize edge computing nodes across regions"""
        
        # Define edge node configurations
        node_configs = [
            {
                'node_id': 'us-east-1',
                'region': 'US East',
                'capabilities': ['ml_inference', 'data_processing', 'real_time_analysis'],
                'max_concurrent_tasks': 50,
                'latency_ms': 10
            },
            {
                'node_id': 'us-west-1',
                'region': 'US West',
                'capabilities': ['ml_inference', 'data_processing', 'behavioral_analysis'],
                'max_concurrent_tasks': 40,
                'latency_ms': 15
            },
            {
                'node_id': 'eu-west-1',
                'region': 'Europe',
                'capabilities': ['data_processing', 'risk_analysis'],
                'max_concurrent_tasks': 30,
                'latency_ms': 25
            }
        ]
        
        # Initialize nodes
        for config in node_configs:
            node = EdgeNode(
                node_id=config['node_id'],
                region=config['region'],
                capabilities=config['capabilities'],
                max_concurrent_tasks=config['max_concurrent_tasks'],
                current_load=0.0,
                latency_ms=config['latency_ms'],
                available=True,
                last_heartbeat=datetime.now()
            )
            
            self.edge_nodes[node.node_id] = node
        
        self.logger.info(f"✅ Initialized {len(self.edge_nodes)} edge nodes")
    
    async def submit_processing_task(self, 
                                   task_type: str,
                                   data: Dict[str, Any],
                                   priority: int = 5) -> str:
        """Submit a processing task to the edge network"""
        
        task_id = self._generate_task_id(task_type, data)
        
        task = {
            'task_id': task_id,
            'task_type': task_type,
            'priority': priority,
            'data': data,
            'created_at': datetime.now(),
            'result': None
        }
        
        # Add to task queue
        await self.task_queue.put(task)
        self.active_tasks[task_id] = task
        
        self.logger.info(f"📋 Task {task_id} submitted: {task_type}")
        return task_id
    
    async def task_distribution_loop(self):
        """Main loop for distributing tasks to edge nodes"""
        while True:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find optimal node for task
                optimal_node = await self.find_optimal_node(task)
                
                if optimal_node:
                    # Execute task on node
                    asyncio.create_task(self.execute_task_on_node(task, optimal_node))
                else:
                    # No available nodes, re-queue
                    await asyncio.sleep(0.1)
                    await self.task_queue.put(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in task distribution: {e}")
    
    async def find_optimal_node(self, task: Dict[str, Any]) -> Optional[str]:
        """Find optimal edge node for task execution"""
        
        best_node = None
        best_score = -1
        
        for node_id, node in self.edge_nodes.items():
            if not node.available or node.current_load > 0.8:
                continue
            
            # Calculate node suitability score
            load_factor = 1.0 - node.current_load
            latency_factor = 1.0 / (1.0 + node.latency_ms / 100)
            
            score = load_factor * 0.6 + latency_factor * 0.4
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        return best_node
    
    async def execute_task_on_node(self, task: Dict[str, Any], node_id: str):
        """Execute task on edge node"""
        
        node = self.edge_nodes[node_id]
        start_time = time.time()
        
        try:
            # Update node load
            node.current_load = min(1.0, node.current_load + 0.1)
            
            # Simulate task execution
            result = await self._simulate_task_execution(task, node)
            
            # Store result
            task['result'] = result
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics['total_tasks_processed'] += 1
            
            # Store result in Redis
            await self.redis_set(
                f"task_result:{task['task_id']}", 
                json.dumps(result), 
                ex=3600
            )
            
            self.logger.info(f"✅ Task {task['task_id']} completed in {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Task {task['task_id']} failed: {e}")
        
        finally:
            # Update node load
            node.current_load = max(0.0, node.current_load - 0.1)
            
            # Remove from active tasks
            if task['task_id'] in self.active_tasks:
                del self.active_tasks[task['task_id']]
    
    async def _simulate_task_execution(self, task: Dict[str, Any], node: EdgeNode) -> Dict[str, Any]:
        """Simulate task execution based on type"""
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        task_type = task['task_type']
        data = task['data']
        
        if task_type == 'ml_inference':
            return {
                'symbol': data.get('symbol', 'UNKNOWN'),
                'prediction': {
                    'direction': np.random.choice(['buy', 'sell', 'hold']),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'magnitude': np.random.uniform(0.02, 0.15)
                },
                'node_id': node.node_id,
                'processing_time_ms': 100
            }
        
        elif task_type == 'behavioral_analysis':
            return {
                'member_id': data.get('member_id', 0),
                'behavioral_score': np.random.uniform(0.6, 0.9),
                'node_id': node.node_id,
                'processing_time_ms': 80
            }
        
        else:
            return {
                'result': 'processed',
                'node_id': node.node_id,
                'processing_time_ms': 50
            }
    
    async def get_task_result(self, task_id: str, timeout_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """Get result of submitted task"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check Redis for result
            result_data = await self.redis_get(f"task_result:{task_id}")
            
            if result_data:
                return json.loads(result_data)
            
            await asyncio.sleep(0.1)
        
        return {'error': 'Task timeout', 'task_id': task_id}
    
    async def node_health_monitoring(self):
        """Monitor health of edge nodes"""
        while True:
            try:
                for node_id, node in self.edge_nodes.items():
                    node.last_heartbeat = datetime.now()
                    node.available = node.current_load < 0.9
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    async def performance_monitoring(self):
        """Monitor system performance metrics"""
        while True:
            try:
                total_load = sum(node.current_load for node in self.edge_nodes.values())
                self.metrics['current_system_load'] = total_load / len(self.edge_nodes)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    def _generate_task_id(self, task_type: str, data: Dict[str, Any]) -> str:
        """Generate unique task ID"""
        timestamp = str(int(time.time() * 1000))
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
        return f"{task_type}_{timestamp}_{data_hash}"
    
    # Redis async wrapper methods
    async def redis_get(self, key: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.redis_client.get, key)
    
    async def redis_set(self, key: str, value: str, ex: int = None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.redis_client.set, key, value, ex)

# Example usage
async def main():
    """Test the Edge Computing Architecture"""
    print("🌐 Testing APEX Edge Computing Architecture")
    
    orchestrator = EdgeComputingOrchestrator()
    
    # Start orchestrator
    orchestrator_task = asyncio.create_task(orchestrator.start_orchestrator())
    
    await asyncio.sleep(2)  # Wait for initialization
    
    # Submit test tasks
    task_ids = []
    for i in range(5):
        task_id = await orchestrator.submit_processing_task(
            task_type='ml_inference',
            data={'symbol': f'TEST{i}', 'features': [1, 2, 3]},
            priority=8
        )
        task_ids.append(task_id)
    
    # Wait and get results
    await asyncio.sleep(3)
    
    completed = 0
    for task_id in task_ids:
        result = await orchestrator.get_task_result(task_id, timeout_seconds=5)
        if result and 'error' not in result:
            completed += 1
    
    print(f"✅ Edge Computing Test: {completed}/{len(task_ids)} tasks completed")
    
    orchestrator_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
