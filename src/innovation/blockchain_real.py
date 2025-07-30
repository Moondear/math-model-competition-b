"""
真实区块链供应链记录模块
使用Web3.py实现智能合约和区块链交互
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

# 尝试导入web3
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    from eth_account import Account
    from solcx import compile_source, install_solc, set_solc_version
    HAS_WEB3 = True
except ImportError as e:
    print(f"警告: Web3依赖导入失败: {e}，使用模拟实现")
    HAS_WEB3 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BlockchainConfig:
    """区块链配置"""
    provider_url: str = "http://localhost:8545"  # Ganache本地节点
    chain_id: int = 1337
    gas_limit: int = 3000000
    gas_price: int = 20000000000  # 20 Gwei
    private_key: Optional[str] = None
    contract_address: Optional[str] = None

@dataclass
class ProductionDecision:
    """生产决策记录"""
    timestamp: int
    case_id: int
    decision_hash: str
    test_part1: bool
    test_part2: bool
    test_final: bool
    repair: bool
    expected_profit: float
    defect_rate1: float
    defect_rate2: float
    validator_address: str
    signature: str

class RealBlockchainManager:
    """真实区块链管理器"""
    
    # 智能合约源码
    CONTRACT_SOURCE = '''
    pragma solidity ^0.8.0;

    contract ProductionDecisionRegistry {
        struct Decision {
            uint256 timestamp;
            uint256 caseId;
            string decisionHash;
            bool testPart1;
            bool testPart2;
            bool testFinal;
            bool repair;
            uint256 expectedProfit;
            uint256 defectRate1;
            uint256 defectRate2;
            address validator;
            string signature;
        }
        
        mapping(uint256 => Decision) public decisions;
        mapping(string => bool) public hashExists;
        uint256 public decisionCount;
        address public owner;
        
        event DecisionRecorded(
            uint256 indexed decisionId,
            uint256 indexed caseId,
            string decisionHash,
            address validator
        );
        
        modifier onlyOwner() {
            require(msg.sender == owner, "Only owner can call this function");
            _;
        }
        
        constructor() {
            owner = msg.sender;
            decisionCount = 0;
        }
        
        function recordDecision(
            uint256 _caseId,
            string memory _decisionHash,
            bool _testPart1,
            bool _testPart2,
            bool _testFinal,
            bool _repair,
            uint256 _expectedProfit,
            uint256 _defectRate1,
            uint256 _defectRate2,
            string memory _signature
        ) public returns (uint256) {
            require(!hashExists[_decisionHash], "Decision hash already exists");
            
            uint256 decisionId = decisionCount;
            
            decisions[decisionId] = Decision({
                timestamp: block.timestamp,
                caseId: _caseId,
                decisionHash: _decisionHash,
                testPart1: _testPart1,
                testPart2: _testPart2,
                testFinal: _testFinal,
                repair: _repair,
                expectedProfit: _expectedProfit,
                defectRate1: _defectRate1,
                defectRate2: _defectRate2,
                validator: msg.sender,
                signature: _signature
            });
            
            hashExists[_decisionHash] = true;
            decisionCount++;
            
            emit DecisionRecorded(decisionId, _caseId, _decisionHash, msg.sender);
            
            return decisionId;
        }
        
        function getDecision(uint256 _decisionId) public view returns (Decision memory) {
            require(_decisionId < decisionCount, "Decision does not exist");
            return decisions[_decisionId];
        }
        
        function verifyDecisionHash(string memory _decisionHash) public view returns (bool) {
            return hashExists[_decisionHash];
        }
        
        function getDecisionCount() public view returns (uint256) {
            return decisionCount;
        }
    }
    '''
    
    def __init__(self, config: BlockchainConfig):
        """初始化区块链管理器
        
        Args:
            config: 区块链配置
        """
        self.config = config
        self.w3 = None
        self.account = None
        self.contract = None
        
        if HAS_WEB3:
            self._initialize_web3()
        else:
            logger.warning("Web3不可用，将使用模拟区块链")
    
    def _initialize_web3(self):
        """初始化Web3连接"""
        try:
            # 连接到区块链节点
            self.w3 = Web3(Web3.HTTPProvider(self.config.provider_url))
            
            # 添加POA中间件（适用于Ganache等测试网络）
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if self.w3.is_connected():
                logger.info(f"已连接到区块链节点: {self.config.provider_url}")
                
                # 设置默认账户
                if self.config.private_key:
                    self.account = Account.from_key(self.config.private_key)
                    self.w3.eth.default_account = self.account.address
                else:
                    # 使用节点的第一个账户
                    accounts = self.w3.eth.accounts
                    if accounts:
                        self.w3.eth.default_account = accounts[0]
                        logger.info(f"使用默认账户: {accounts[0]}")
                
                # 部署或连接合约
                if self.config.contract_address:
                    self._connect_to_contract()
                else:
                    self._deploy_contract()
                    
            else:
                logger.error("无法连接到区块链节点")
                self.w3 = None
                
        except Exception as e:
            logger.error(f"Web3初始化失败: {e}")
            self.w3 = None
    
    def _deploy_contract(self) -> str:
        """部署智能合约
        
        Returns:
            str: 合约地址
        """
        logger.info("开始部署智能合约...")
        
        try:
            # 安装和设置Solidity编译器
            try:
                install_solc('0.8.19')
                set_solc_version('0.8.19')
            except:
                logger.warning("Solidity编译器设置失败，尝试使用默认版本")
            
            # 编译合约
            compiled_sol = compile_source(self.CONTRACT_SOURCE)
            contract_interface = compiled_sol['<stdin>:ProductionDecisionRegistry']
            
            # 部署合约
            contract = self.w3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bin']
            )
            
            # 构建交易
            transaction = contract.constructor().build_transaction({
                'chainId': self.config.chain_id,
                'gas': self.config.gas_limit,
                'gasPrice': self.config.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            })
            
            # 签名和发送交易
            if self.account:
                signed_txn = self.w3.eth.account.sign_transaction(transaction, self.config.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            else:
                tx_hash = self.w3.eth.send_transaction(transaction)
            
            # 等待交易确认
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # 创建合约实例
            self.contract = self.w3.eth.contract(
                address=tx_receipt.contractAddress,
                abi=contract_interface['abi']
            )
            
            self.config.contract_address = tx_receipt.contractAddress
            
            logger.info(f"合约部署成功，地址: {tx_receipt.contractAddress}")
            logger.info(f"Gas 消耗: {tx_receipt.gasUsed}")
            
            return tx_receipt.contractAddress
            
        except Exception as e:
            logger.error(f"合约部署失败: {e}")
            raise
    
    def _connect_to_contract(self):
        """连接到已部署的合约"""
        try:
            # 这里需要合约ABI，实际使用时应该保存ABI
            # 为简化，我们重新编译获取ABI
            compiled_sol = compile_source(self.CONTRACT_SOURCE)
            contract_interface = compiled_sol['<stdin>:ProductionDecisionRegistry']
            
            self.contract = self.w3.eth.contract(
                address=self.config.contract_address,
                abi=contract_interface['abi']
            )
            
            logger.info(f"已连接到合约: {self.config.contract_address}")
            
        except Exception as e:
            logger.error(f"连接合约失败: {e}")
            raise
    
    def record_production_decision(self, decision_data: Dict) -> Dict:
        """记录生产决策到区块链
        
        Args:
            decision_data: 决策数据
            
        Returns:
            Dict: 区块链记录结果
        """
        logger.info("开始记录生产决策到区块链...")
        
        if not HAS_WEB3 or not self.w3 or not self.contract:
            return self._simulate_blockchain_record(decision_data)
        
        try:
            # 创建决策哈希
            decision_hash = self._create_decision_hash(decision_data)
            
            # 创建数字签名
            signature = self._sign_decision(decision_data, decision_hash)
            
            # 准备合约调用参数
            case_id = decision_data.get('case_id', 1)
            test_part1 = decision_data.get('test_part1', False)
            test_part2 = decision_data.get('test_part2', False) 
            test_final = decision_data.get('test_final', False)
            repair = decision_data.get('repair', False)
            expected_profit = int(decision_data.get('expected_profit', 0) * 100)  # 转换为整数(分)
            defect_rate1 = int(decision_data.get('defect_rate1', 0) * 10000)  # 转换为基点
            defect_rate2 = int(decision_data.get('defect_rate2', 0) * 10000)
            
            # 构建交易
            transaction = self.contract.functions.recordDecision(
                case_id,
                decision_hash,
                test_part1,
                test_part2,
                test_final,
                repair,
                expected_profit,
                defect_rate1,
                defect_rate2,
                signature
            ).build_transaction({
                'chainId': self.config.chain_id,
                'gas': self.config.gas_limit,
                'gasPrice': self.config.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            })
            
            # 签名和发送交易
            if self.account:
                signed_txn = self.w3.eth.account.sign_transaction(transaction, self.config.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            else:
                tx_hash = self.w3.eth.send_transaction(transaction)
            
            # 等待交易确认
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # 解析事件日志
            decision_events = self.contract.events.DecisionRecorded().process_receipt(tx_receipt)
            decision_id = decision_events[0]['args']['decisionId'] if decision_events else None
            
            return {
                'success': True,
                'transaction_hash': tx_hash.hex(),
                'block_number': tx_receipt.blockNumber,
                'block_hash': tx_receipt.blockHash.hex(),
                'decision_id': decision_id,
                'decision_hash': decision_hash,
                'gas_used': tx_receipt.gasUsed,
                'gas_price': self.config.gas_price,
                'total_cost': tx_receipt.gasUsed * self.config.gas_price / 10**18,  # ETH
                'contract_address': self.config.contract_address,
                'validator_address': self.w3.eth.default_account,
                'timestamp': int(time.time()),
                'signature': signature
            }
            
        except Exception as e:
            logger.error(f"区块链记录失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_hash': self._create_decision_hash(decision_data)
            }
    
    def verify_decision_integrity(self, decision_id: int) -> Dict:
        """验证决策记录的完整性
        
        Args:
            decision_id: 决策ID
            
        Returns:
            Dict: 验证结果
        """
        logger.info(f"验证决策记录完整性: {decision_id}")
        
        if not HAS_WEB3 or not self.w3 or not self.contract:
            return self._simulate_verification(decision_id)
        
        try:
            # 从区块链获取决策记录
            decision = self.contract.functions.getDecision(decision_id).call()
            
            # 重构决策数据
            decision_data = {
                'case_id': decision[1],
                'test_part1': decision[3],
                'test_part2': decision[4],
                'test_final': decision[5],
                'repair': decision[6],
                'expected_profit': decision[7] / 100.0,
                'defect_rate1': decision[8] / 10000.0,
                'defect_rate2': decision[9] / 10000.0
            }
            
            # 重新计算哈希
            computed_hash = self._create_decision_hash(decision_data)
            stored_hash = decision[2]
            
            # 验证哈希一致性
            hash_valid = computed_hash == stored_hash
            
            # 验证签名
            signature_valid = self._verify_signature(decision_data, stored_hash, decision[11])
            
            return {
                'decision_id': decision_id,
                'hash_valid': hash_valid,
                'signature_valid': signature_valid,
                'integrity_score': (hash_valid + signature_valid) / 2,
                'stored_hash': stored_hash,
                'computed_hash': computed_hash,
                'timestamp': decision[0],
                'validator': decision[10],
                'block_verified': True
            }
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return {
                'decision_id': decision_id,
                'hash_valid': False,
                'signature_valid': False,
                'integrity_score': 0.0,
                'error': str(e)
            }
    
    def get_supply_chain_audit_trail(self, start_block: int = 0) -> List[Dict]:
        """获取供应链审计追踪
        
        Args:
            start_block: 起始区块号
            
        Returns:
            List[Dict]: 审计记录列表
        """
        logger.info("获取供应链审计追踪...")
        
        if not HAS_WEB3 or not self.w3 or not self.contract:
            return self._simulate_audit_trail()
        
        try:
            # 获取所有DecisionRecorded事件
            event_filter = self.contract.events.DecisionRecorded.create_filter(
                fromBlock=start_block,
                toBlock='latest'
            )
            
            events = event_filter.get_all_entries()
            audit_trail = []
            
            for event in events:
                # 获取详细的决策记录
                decision_id = event['args']['decisionId']
                decision = self.contract.functions.getDecision(decision_id).call()
                
                # 获取交易详情
                tx_hash = event['transactionHash']
                tx = self.w3.eth.get_transaction(tx_hash)
                block = self.w3.eth.get_block(event['blockNumber'])
                
                audit_record = {
                    'decision_id': decision_id,
                    'case_id': decision[1],
                    'decision_hash': decision[2],
                    'timestamp': decision[0],
                    'block_number': event['blockNumber'],
                    'block_hash': event['blockHash'].hex(),
                    'transaction_hash': tx_hash.hex(),
                    'validator_address': decision[10],
                    'gas_used': tx.get('gas', 0),
                    'gas_price': tx.get('gasPrice', 0),
                    'block_timestamp': block['timestamp'],
                    'expected_profit': decision[7] / 100.0,
                    'defect_rates': [decision[8] / 10000.0, decision[9] / 10000.0],
                    'decisions': {
                        'test_part1': decision[3],
                        'test_part2': decision[4],
                        'test_final': decision[5],
                        'repair': decision[6]
                    }
                }
                
                audit_trail.append(audit_record)
            
            logger.info(f"获取到 {len(audit_trail)} 条审计记录")
            return audit_trail
            
        except Exception as e:
            logger.error(f"获取审计追踪失败: {e}")
            return []
    
    def _create_decision_hash(self, decision_data: Dict) -> str:
        """创建决策数据哈希
        
        Args:
            decision_data: 决策数据
            
        Returns:
            str: 决策哈希
        """
        # 创建标准化的数据字符串
        data_string = (
            f"{decision_data.get('case_id', 0)}"
            f"{decision_data.get('test_part1', False)}"
            f"{decision_data.get('test_part2', False)}"
            f"{decision_data.get('test_final', False)}"
            f"{decision_data.get('repair', False)}"
            f"{decision_data.get('expected_profit', 0):.2f}"
            f"{decision_data.get('defect_rate1', 0):.4f}"
            f"{decision_data.get('defect_rate2', 0):.4f}"
            f"{int(time.time())}"
        )
        
        # 计算SHA-256哈希
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _sign_decision(self, decision_data: Dict, decision_hash: str) -> str:
        """对决策进行数字签名
        
        Args:
            decision_data: 决策数据
            decision_hash: 决策哈希
            
        Returns:
            str: 数字签名
        """
        if self.account and self.config.private_key:
            # 创建消息哈希
            message = f"ProductionDecision:{decision_hash}:{decision_data.get('case_id', 0)}"
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            
            # 使用私钥签名
            signature = Account.sign_message_hash(bytes.fromhex(message_hash), self.config.private_key)
            return signature.signature.hex()
        else:
            # 模拟签名
            return hashlib.md5(f"{decision_hash}{time.time()}".encode()).hexdigest()
    
    def _verify_signature(self, decision_data: Dict, decision_hash: str, signature: str) -> bool:
        """验证数字签名
        
        Args:
            decision_data: 决策数据
            decision_hash: 决策哈希
            signature: 数字签名
            
        Returns:
            bool: 签名是否有效
        """
        try:
            if self.account:
                message = f"ProductionDecision:{decision_hash}:{decision_data.get('case_id', 0)}"
                message_hash = hashlib.sha256(message.encode()).hexdigest()
                
                # 恢复签名者地址
                recovered_address = Account.recover_message_hash(
                    bytes.fromhex(message_hash), 
                    signature=bytes.fromhex(signature)
                )
                
                return recovered_address.lower() == self.account.address.lower()
            else:
                # 模拟验证
                return len(signature) == 32  # 简单长度检查
                
        except Exception as e:
            logger.error(f"签名验证失败: {e}")
            return False
    
    def _simulate_blockchain_record(self, decision_data: Dict) -> Dict:
        """模拟区块链记录（当Web3不可用时）"""
        decision_hash = self._create_decision_hash(decision_data)
        
        return {
            'success': True,
            'transaction_hash': f"0x{hashlib.sha256(f'{decision_hash}{time.time()}'.encode()).hexdigest()}",
            'block_number': int(time.time()) % 1000000,
            'decision_hash': decision_hash,
            'gas_used': 200000,
            'total_cost': 0.004,  # 模拟成本
            'timestamp': int(time.time()),
            'simulation': True
        }
    
    def _simulate_verification(self, decision_id: int) -> Dict:
        """模拟验证（当Web3不可用时）"""
        return {
            'decision_id': decision_id,
            'hash_valid': True,
            'signature_valid': True,
            'integrity_score': 1.0,
            'simulation': True
        }
    
    def _simulate_audit_trail(self) -> List[Dict]:
        """模拟审计追踪（当Web3不可用时）"""
        return [
            {
                'decision_id': i,
                'case_id': (i % 6) + 1,
                'timestamp': int(time.time()) - i * 3600,
                'block_number': 1000000 + i,
                'simulation': True
            }
            for i in range(5)
        ]


if __name__ == "__main__":
    # 测试真实区块链管理器
    config = BlockchainConfig(
        provider_url="http://localhost:8545",  # 需要运行Ganache
        gas_limit=3000000,
        gas_price=20000000000
    )
    
    blockchain = RealBlockchainManager(config)
    
    # 测试记录决策
    decision_data = {
        'case_id': 1,
        'test_part1': True,
        'test_part2': True,
        'test_final': False,
        'repair': True,
        'expected_profit': 45.5,
        'defect_rate1': 0.1,
        'defect_rate2': 0.1
    }
    
    result = blockchain.record_production_decision(decision_data)
    print("区块链记录结果:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    # 测试验证
    if result.get('success') and result.get('decision_id') is not None:
        verification = blockchain.verify_decision_integrity(result['decision_id'])
        print(f"\n验证结果:")
        for key, value in verification.items():
            print(f"{key}: {value}")
    
    # 测试审计追踪
    audit_trail = blockchain.get_supply_chain_audit_trail()
    print(f"\n审计追踪记录数: {len(audit_trail)}") 