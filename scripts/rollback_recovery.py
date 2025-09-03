#!/usr/bin/env python3
"""
IFRS9 Pipeline Rollback and Recovery System

This script provides comprehensive rollback and recovery procedures for the IFRS9
pipeline, ensuring data integrity and system stability in case of failures.
"""

import json
import logging
import shutil
import sqlite3
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import psutil
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/airflow/logs/rollback_recovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for pipeline data backups."""
    backup_id: str
    pipeline_id: str
    backup_timestamp: str
    backup_type: str  # full, incremental, configuration
    backup_size_mb: float
    backup_path: str
    checksum: str
    retention_days: int
    status: str  # active, archived, deleted


@dataclass
class RollbackAction:
    """Individual rollback action record."""
    action_id: str
    action_type: str  # file_restore, database_rollback, configuration_revert
    target_path: str
    backup_source: str
    execution_timestamp: str
    status: str  # pending, executed, failed
    error_message: Optional[str] = None


class RollbackRecoveryManager:
    """
    Comprehensive rollback and recovery manager for IFRS9 pipeline.
    
    Provides:
    - Automated backup creation before pipeline execution
    - Point-in-time recovery capabilities
    - Configuration rollback procedures
    - Data integrity verification
    - System state restoration
    """
    
    def __init__(self, config_path: str = "/opt/airflow/config/orchestration_rules.yaml"):
        """Initialize the rollback recovery manager."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize backup tracking database
        self.backup_db_path = "/opt/airflow/data/backup_metadata.db"
        self._initialize_backup_database()
        
        # Define critical paths for backup
        self.critical_paths = [
            "/opt/airflow/data/raw",
            "/opt/airflow/data/processed", 
            "/opt/airflow/models",
            "/opt/airflow/config",
            "/opt/airflow/logs"
        ]
        
        # Define backup storage location
        self.backup_root = Path("/opt/airflow/backups")
        self.backup_root.mkdir(exist_ok=True)
        
        logger.info("Rollback Recovery Manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load orchestration configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return {'operational_settings': {'backup_recovery': {'backup_retention_days': 30}}}
    
    def _initialize_backup_database(self):
        """Initialize SQLite database for backup metadata tracking."""
        with sqlite3.connect(self.backup_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backups (
                    backup_id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    backup_timestamp TEXT,
                    backup_type TEXT,
                    backup_size_mb REAL,
                    backup_path TEXT,
                    checksum TEXT,
                    retention_days INTEGER,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rollback_actions (
                    action_id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    action_type TEXT,
                    target_path TEXT,
                    backup_source TEXT,
                    execution_timestamp TEXT,
                    status TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        logger.info("Backup database initialized")
    
    def create_pipeline_backup(self, pipeline_id: str, backup_type: str = "full") -> BackupMetadata:
        """
        Create a comprehensive backup before pipeline execution.
        
        Args:
            pipeline_id: Unique identifier for the pipeline run
            backup_type: Type of backup (full, incremental, configuration)
            
        Returns:
            BackupMetadata: Metadata about the created backup
        """
        backup_timestamp = datetime.now()
        backup_id = f"BACKUP_{pipeline_id}_{backup_timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating {backup_type} backup: {backup_id}")
        
        # Create backup directory
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(exist_ok=True)
        
        total_size_mb = 0.0
        backup_manifest = {
            'backup_id': backup_id,
            'pipeline_id': pipeline_id,
            'backup_type': backup_type,
            'backup_timestamp': backup_timestamp.isoformat(),
            'backed_up_paths': [],
            'file_checksums': {},
            'system_state': self._capture_system_state()
        }
        
        try:
            # Backup critical paths
            for critical_path in self.critical_paths:
                if Path(critical_path).exists():
                    backup_target = backup_dir / Path(critical_path).name
                    
                    if Path(critical_path).is_dir():\n                        shutil.copytree(critical_path, backup_target, ignore_dangling_symlinks=True)
                    else:
                        shutil.copy2(critical_path, backup_target)
                    \n                    # Calculate size and checksum\n                    path_size = self._calculate_path_size(backup_target)\n                    path_checksum = self._calculate_path_checksum(backup_target)\n                    \n                    backup_manifest['backed_up_paths'].append({\n                        'original_path': critical_path,\n                        'backup_path': str(backup_target),\n                        'size_mb': path_size,\n                        'checksum': path_checksum\n                    })\n                    \n                    backup_manifest['file_checksums'][critical_path] = path_checksum\n                    total_size_mb += path_size\n            \n            # Save backup manifest\n            manifest_path = backup_dir / 'backup_manifest.json'\n            with open(manifest_path, 'w') as f:\n                json.dump(backup_manifest, f, indent=2, default=str)\n            \n            # Calculate overall backup checksum\n            backup_checksum = self._calculate_path_checksum(backup_dir)\n            \n            # Create backup metadata\n            retention_days = self.config.get('operational_settings', {}).get('backup_recovery', {}).get('backup_retention_days', 30)\n            \n            backup_metadata = BackupMetadata(\n                backup_id=backup_id,\n                pipeline_id=pipeline_id,\n                backup_timestamp=backup_timestamp.isoformat(),\n                backup_type=backup_type,\n                backup_size_mb=total_size_mb,\n                backup_path=str(backup_dir),\n                checksum=backup_checksum,\n                retention_days=retention_days,\n                status='active'\n            )\n            \n            # Store backup metadata in database\n            self._store_backup_metadata(backup_metadata)\n            \n            logger.info(f"Backup created successfully: {backup_id} ({total_size_mb:.2f} MB)")\n            \n            return backup_metadata\n            \n        except Exception as e:\n            logger.error(f"Backup creation failed: {str(e)}")ß\n            \n            # Cleanup failed backup\n            if backup_dir.exists():\n                shutil.rmtree(backup_dir)\n            \n            raise Exception(f"Backup creation failed: {str(e)}")\n    \n    def _capture_system_state(self) -> Dict[str, Any]:\n        """Capture current system state for recovery purposes."""\n        try:\n            system_state = {\n                'timestamp': datetime.now().isoformat(),\n                'cpu_count': psutil.cpu_count(),\n                'memory_total_gb': psutil.virtual_memory().total / (1024**3),\n                'disk_usage': {\n                    '/opt/airflow': {\n                        'total_gb': psutil.disk_usage('/opt/airflow').total / (1024**3),\n                        'used_gb': psutil.disk_usage('/opt/airflow').used / (1024**3),\n                        'free_gb': psutil.disk_usage('/opt/airflow').free / (1024**3)\n                    }\n                },\n                'running_processes': len(psutil.pids()),\n                'network_interfaces': list(psutil.net_if_addrs().keys()),\n                'python_version': subprocess.check_output(['python', '--version'], text=True).strip(),\n                'airflow_version': self._get_airflow_version()\n            }\n            \n            return system_state\n            \n        except Exception as e:\n            logger.warning(f"Could not capture complete system state: {e}")\n            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}\n    \n    def _get_airflow_version(self) -> str:\n        """Get Airflow version."""\n        try:\n            result = subprocess.check_output(['airflow', 'version'], text=True)\n            return result.strip()\n        except:\n            return 'unknown'\n    \n    def _calculate_path_size(self, path: Path) -> float:\n        """Calculate size of a file or directory in MB."""\n        if path.is_file():\n            return path.stat().st_size / (1024 * 1024)\n        elif path.is_dir():\n            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())\n            return total_size / (1024 * 1024)\n        return 0.0\n    \n    def _calculate_path_checksum(self, path: Path) -> str:\n        """Calculate MD5 checksum for a file or directory."""\n        hash_md5 = hashlib.md5()\n        \n        if path.is_file():\n            with open(path, 'rb') as f:\n                for chunk in iter(lambda: f.read(4096), b""):\n                    hash_md5.update(chunk)\n        elif path.is_dir():\n            # For directories, create checksum based on file structure and content\n            for file_path in sorted(path.rglob('*')):\n                if file_path.is_file():\n                    # Include relative path in hash\n                    hash_md5.update(str(file_path.relative_to(path)).encode())\n                    with open(file_path, 'rb') as f:\n                        for chunk in iter(lambda: f.read(4096), b""):\n                            hash_md5.update(chunk)\n        \n        return hash_md5.hexdigest()\n    \n    def _store_backup_metadata(self, metadata: BackupMetadata):\n        """Store backup metadata in database."""\n        with sqlite3.connect(self.backup_db_path) as conn:\n            conn.execute(\"\"\"\n                INSERT INTO backups \n                (backup_id, pipeline_id, backup_timestamp, backup_type, backup_size_mb, \n                 backup_path, checksum, retention_days, status) \n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)\n            \"\"\", (\n                metadata.backup_id, metadata.pipeline_id, metadata.backup_timestamp,\n                metadata.backup_type, metadata.backup_size_mb, metadata.backup_path,\n                metadata.checksum, metadata.retention_days, metadata.status\n            ))\n            conn.commit()\n    \n    def execute_rollback(self, backup_id: str, target_paths: Optional[List[str]] = None) -> Dict[str, Any]:\n        """\n        Execute rollback to a specific backup point.\n        \n        Args:\n            backup_id: ID of the backup to restore from\n            target_paths: Specific paths to restore (None for full restore)\n            \n        Returns:\n            Dict containing rollback results and actions taken\n        """\n        logger.info(f"Starting rollback to backup: {backup_id}")n        \n        # Get backup metadata\n        backup_metadata = self._get_backup_metadata(backup_id)\n        if not backup_metadata:\n            raise Exception(f"Backup not found: {backup_id}")\n        \n        rollback_timestamp = datetime.now()\n        rollback_actions: List[RollbackAction] = []\n        rollback_summary = {\n            'backup_id': backup_id,\n            'rollback_timestamp': rollback_timestamp.isoformat(),\n            'actions_attempted': 0,\n            'actions_successful': 0,\n            'actions_failed': 0,\n            'errors': [],\n            'restored_paths': []\n        }\n        \n        try:\n            # Load backup manifest\n            backup_path = Path(backup_metadata['backup_path'])\n            manifest_path = backup_path / 'backup_manifest.json'\n            \n            with open(manifest_path, 'r') as f:\n                backup_manifest = json.load(f)\n            \n            # Determine paths to restore\n            paths_to_restore = target_paths or [item['original_path'] for item in backup_manifest['backed_up_paths']]\n            \n            # Execute rollback actions\n            for original_path in paths_to_restore:\n                action = self._execute_rollback_action(\n                    backup_manifest, \n                    original_path, \n                    backup_metadata['pipeline_id'],\n                    rollback_timestamp\n                )\n                \n                rollback_actions.append(action)\n                rollback_summary['actions_attempted'] += 1\n                \n                if action.status == 'executed':\n                    rollback_summary['actions_successful'] += 1\n                    rollback_summary['restored_paths'].append(original_path)\n                else:\n                    rollback_summary['actions_failed'] += 1\n                    if action.error_message:\n                        rollback_summary['errors'].append({\n                            'path': original_path,\n                            'error': action.error_message\n                        })\n            \n            # Verify rollback integrity\n            integrity_check = self._verify_rollback_integrity(backup_manifest, paths_to_restore)\n            rollback_summary['integrity_check'] = integrity_check\n            \n            # Store rollback actions in database\n            for action in rollback_actions:\n                self._store_rollback_action(action)\n            \n            # Save rollback summary\n            rollback_summary_path = f"/opt/airflow/data/processed/rollback_summary_{backup_id}_{int(rollback_timestamp.timestamp())}.json"\n            with open(rollback_summary_path, 'w') as f:\n                json.dump(rollback_summary, f, indent=2, default=str)\n            \n            success_rate = (rollback_summary['actions_successful'] / rollback_summary['actions_attempted']) * 100 if rollback_summary['actions_attempted'] > 0 else 0\n            \n            logger.info(f"Rollback completed - Success rate: {success_rate:.1f}% ({rollback_summary['actions_successful']}/{rollback_summary['actions_attempted']})")\n            \n            return rollback_summary\n            \n        except Exception as e:\n            logger.error(f"Rollback execution failed: {str(e)}")\n            rollback_summary['errors'].append({'general_error': str(e)})\n            return rollback_summary\n    \n    def _get_backup_metadata(self, backup_id: str) -> Optional[Dict[str, Any]]:\n        """Retrieve backup metadata from database."""\n        with sqlite3.connect(self.backup_db_path) as conn:\n            cursor = conn.execute(\n                "SELECT * FROM backups WHERE backup_id = ? AND status = 'active'", \n                (backup_id,)\n            )\n            \n            row = cursor.fetchone()\n            if row:\n                columns = [description[0] for description in cursor.description]\n                return dict(zip(columns, row))\n            \n            return None\n    \n    def _execute_rollback_action(self, backup_manifest: Dict[str, Any], original_path: str, pipeline_id: str, rollback_timestamp: datetime) -> RollbackAction:\n        """Execute a single rollback action."""\n        action_id = f"ROLLBACK_{pipeline_id}_{int(rollback_timestamp.timestamp())}_{len(original_path)}"\n        \n        action = RollbackAction(\n            action_id=action_id,\n            action_type='file_restore',\n            target_path=original_path,\n            backup_source='',\n            execution_timestamp=rollback_timestamp.isoformat(),\n            status='pending'\n        )\n        \n        try:\n            # Find backup info for this path\n            backup_info = None\n            for item in backup_manifest['backed_up_paths']:\n                if item['original_path'] == original_path:\n                    backup_info = item\n                    break\n            \n            if not backup_info:\n                action.status = 'failed'\n                action.error_message = f"No backup found for path: {original_path}"\n                return action\n            \n            action.backup_source = backup_info['backup_path']\n            \n            # Remove existing path if it exists\n            target_path = Path(original_path)\n            if target_path.exists():\n                if target_path.is_dir():\n                    shutil.rmtree(target_path)\n                else:\n                    target_path.unlink()\n            \n            # Restore from backup\n            backup_source = Path(backup_info['backup_path'])\n            if backup_source.is_dir():\n                shutil.copytree(backup_source, target_path)\n            else:\n                # Ensure parent directory exists\n                target_path.parent.mkdir(parents=True, exist_ok=True)\n                shutil.copy2(backup_source, target_path)\n            \n            # Verify restoration\n            if target_path.exists():\n                restored_checksum = self._calculate_path_checksum(target_path)\n                expected_checksum = backup_info['checksum']\n                \n                if restored_checksum == expected_checksum:\n                    action.status = 'executed'\n                    logger.info(f"Successfully restored: {original_path}")\n                else:\n                    action.status = 'failed'\n                    action.error_message = f"Checksum mismatch after restoration: {original_path}"\n            else:\n                action.status = 'failed'\n                action.error_message = f"Path does not exist after restoration: {original_path}"\n        \n        except Exception as e:\n            action.status = 'failed'\n            action.error_message = str(e)\n            logger.error(f"Rollback action failed for {original_path}: {str(e)}")\n        \n        return action\n    \n    def _verify_rollback_integrity(self, backup_manifest: Dict[str, Any], restored_paths: List[str]) -> Dict[str, Any]:\n        """Verify the integrity of restored files."""\n        integrity_results = {\n            'verification_timestamp': datetime.now().isoformat(),\n            'paths_verified': 0,\n            'paths_valid': 0,\n            'paths_invalid': 0,\n            'checksum_mismatches': [],\n            'missing_paths': []\n        }\n        \n        for original_path in restored_paths:\n            integrity_results['paths_verified'] += 1\n            \n            # Find expected checksum\n            expected_checksum = backup_manifest['file_checksums'].get(original_path)\n            if not expected_checksum:\n                continue\n            \n            # Check if path exists\n            target_path = Path(original_path)\n            if not target_path.exists():\n                integrity_results['missing_paths'].append(original_path)\n                integrity_results['paths_invalid'] += 1\n                continue\n            \n            # Calculate current checksum\n            current_checksum = self._calculate_path_checksum(target_path)\n            \n            if current_checksum == expected_checksum:\n                integrity_results['paths_valid'] += 1\n            else:\n                integrity_results['paths_invalid'] += 1\n                integrity_results['checksum_mismatches'].append({\n                    'path': original_path,\n                    'expected': expected_checksum,\n                    'actual': current_checksum\n                })\n        \n        integrity_results['overall_valid'] = integrity_results['paths_invalid'] == 0\n        \n        return integrity_results\n    \n    def _store_rollback_action(self, action: RollbackAction):\n        """Store rollback action in database."""\n        with sqlite3.connect(self.backup_db_path) as conn:\n            conn.execute(\"\"\"\n                INSERT INTO rollback_actions \n                (action_id, pipeline_id, action_type, target_path, backup_source, \n                 execution_timestamp, status, error_message) \n                VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n            \"\"\", (\n                action.action_id, '', action.action_type, action.target_path,\n                action.backup_source, action.execution_timestamp, \n                action.status, action.error_message\n            ))\n            conn.commit()\n    \n    def list_available_backups(self, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:\n        """List all available backups, optionally filtered by pipeline ID."""\n        with sqlite3.connect(self.backup_db_path) as conn:\n            if pipeline_id:\n                cursor = conn.execute(\n                    "SELECT * FROM backups WHERE pipeline_id = ? AND status = 'active' ORDER BY backup_timestamp DESC",\n                    (pipeline_id,)\n                )\n            else:\n                cursor = conn.execute(\n                    "SELECT * FROM backups WHERE status = 'active' ORDER BY backup_timestamp DESC"\n                )\n            \n            columns = [description[0] for description in cursor.description]\n            return [dict(zip(columns, row)) for row in cursor.fetchall()]\n    \n    def cleanup_old_backups(self) -> Dict[str, Any]:\n        """Clean up backups that have exceeded their retention period."""\n        logger.info("Starting backup cleanup process")\n        \n        cleanup_results = {\n            'cleanup_timestamp': datetime.now().isoformat(),\n            'backups_checked': 0,\n            'backups_deleted': 0,\n            'space_freed_mb': 0.0,\n            'errors': []\n        }\n        \n        with sqlite3.connect(self.backup_db_path) as conn:\n            cursor = conn.execute(\n                "SELECT * FROM backups WHERE status = 'active'"\n            )\n            \n            columns = [description[0] for description in cursor.description]\n            backups = [dict(zip(columns, row)) for row in cursor.fetchall()]\n        \n        for backup in backups:\n            cleanup_results['backups_checked'] += 1\n            \n            backup_timestamp = datetime.fromisoformat(backup['backup_timestamp'])\n            retention_days = backup['retention_days']\n            \n            if datetime.now() - backup_timestamp > timedelta(days=retention_days):\n                try:\n                    # Delete backup directory\n                    backup_path = Path(backup['backup_path'])\n                    if backup_path.exists():\n                        cleanup_results['space_freed_mb'] += self._calculate_path_size(backup_path)\n                        shutil.rmtree(backup_path)\n                    \n                    # Update database status\n                    with sqlite3.connect(self.backup_db_path) as conn:\n                        conn.execute(\n                            "UPDATE backups SET status = 'deleted' WHERE backup_id = ?",\n                            (backup['backup_id'],)\n                        )\n                        conn.commit()\n                    \n                    cleanup_results['backups_deleted'] += 1\n                    logger.info(f"Deleted expired backup: {backup['backup_id']}")\n                    \n                except Exception as e:\n                    error_msg = f"Failed to delete backup {backup['backup_id']}: {str(e)}"\n                    cleanup_results['errors'].append(error_msg)\n                    logger.error(error_msg)\n        \n        logger.info(f"Cleanup completed - Deleted {cleanup_results['backups_deleted']} backups, freed {cleanup_results['space_freed_mb']:.2f} MB")\n        \n        return cleanup_results\n    \n    def create_recovery_plan(self, pipeline_id: str, failure_point: str) -> Dict[str, Any]:\n        """Create a comprehensive recovery plan for a failed pipeline."""\n        recovery_plan = {\n            'plan_id': f"RECOVERY_{pipeline_id}_{int(datetime.now().timestamp())}",\n            'pipeline_id': pipeline_id,\n            'failure_point': failure_point,\n            'plan_timestamp': datetime.now().isoformat(),\n            'recovery_steps': [],\n            'estimated_duration_minutes': 0,\n            'prerequisites': [],\n            'risks': [],\n            'success_criteria': []\n        }\n        \n        # Find relevant backups\n        available_backups = self.list_available_backups(pipeline_id)\n        \n        if available_backups:\n            latest_backup = available_backups[0]\n            \n            # Define recovery steps based on failure point\n            if failure_point in ['data_validation', 'data_generation']:\n                recovery_plan['recovery_steps'] = [\n                    {'step': 1, 'action': 'Restore raw data from backup', 'duration_minutes': 5},\n                    {'step': 2, 'action': 'Verify data integrity', 'duration_minutes': 2},\n                    {'step': 3, 'action': 'Restart pipeline from data validation', 'duration_minutes': 15}\n                ]\n                recovery_plan['estimated_duration_minutes'] = 22\n                \n            elif failure_point in ['ifrs9_processing', 'ml_training']:\n                recovery_plan['recovery_steps'] = [\n                    {'step': 1, 'action': 'Restore processed data from backup', 'duration_minutes': 10},\n                    {'step': 2, 'action': 'Restore model artifacts from backup', 'duration_minutes': 5},\n                    {'step': 3, 'action': 'Verify system resources', 'duration_minutes': 3},\n                    {'step': 4, 'action': 'Restart pipeline from failed stage', 'duration_minutes': 45}\n                ]\n                recovery_plan['estimated_duration_minutes'] = 63\n                \n            elif failure_point in ['report_generation', 'data_upload']:\n                recovery_plan['recovery_steps'] = [\n                    {'step': 1, 'action': 'Verify processed data integrity', 'duration_minutes': 3},\n                    {'step': 2, 'action': 'Check external system connectivity', 'duration_minutes': 2},\n                    {'step': 3, 'action': 'Restart pipeline from failed stage', 'duration_minutes': 20}\n                ]\n                recovery_plan['estimated_duration_minutes'] = 25\n            \n            recovery_plan['backup_reference'] = {\n                'backup_id': latest_backup['backup_id'],\n                'backup_timestamp': latest_backup['backup_timestamp'],\n                'backup_size_mb': latest_backup['backup_size_mb']\n            }\n        else:\n            recovery_plan['recovery_steps'] = [\n                {'step': 1, 'action': 'No backup available - perform full pipeline restart', 'duration_minutes': 150}\n            ]\n            recovery_plan['estimated_duration_minutes'] = 150\n        \n        # Define common prerequisites and risks\n        recovery_plan['prerequisites'] = [\n            'Verify system resources availability',\n            'Ensure external system connectivity',\n            'Confirm data source accessibility',\n            'Check Airflow scheduler status'\n        ]\n        \n        recovery_plan['risks'] = [\n            'Data inconsistency during recovery',\n            'Extended downtime impacting SLAs',\n            'Potential data loss if backup is incomplete',\n            'Resource contention during recovery'\n        ]\n        \n        recovery_plan['success_criteria'] = [\n            'All data integrity checks pass',\n            'Pipeline completes without errors',\n            'Recovery time within estimated duration',\n            'No data loss or corruption detected'\n        ]\n        \n        # Save recovery plan\n        plan_file = f"/opt/airflow/data/processed/recovery_plan_{recovery_plan['plan_id']}.json"\n        with open(plan_file, 'w') as f:\n            json.dump(recovery_plan, f, indent=2, default=str)\n        \n        logger.info(f"Recovery plan created: {recovery_plan['plan_id']} (estimated duration: {recovery_plan['estimated_duration_minutes']} minutes)")\n        \n        return recovery_plan


def main():\n    """Main function to demonstrate rollback and recovery capabilities."""\n    manager = RollbackRecoveryManager()\n    \n    # Demo: Create a backup\n    demo_pipeline_id = f"DEMO_{int(datetime.now().timestamp())}"\n    logger.info(f"Creating demo backup for pipeline: {demo_pipeline_id}")\n    \n    try:\n        backup_metadata = manager.create_pipeline_backup(demo_pipeline_id)\n        logger.info(f"Demo backup created: {backup_metadata.backup_id}")\n        \n        # List available backups\n        backups = manager.list_available_backups()\n        logger.info(f"Found {len(backups)} available backups")\n        \n        # Create recovery plan\n        recovery_plan = manager.create_recovery_plan(demo_pipeline_id, \"data_validation\")\n        logger.info(f"Recovery plan created with {len(recovery_plan['recovery_steps'])} steps")\n        \n        # Cleanup old backups\n        cleanup_results = manager.cleanup_old_backups()\n        logger.info(f"Cleanup completed: {cleanup_results['backups_deleted']} backups deleted")\n        \n    except Exception as e:\n        logger.error(f"Demo failed: {str(e)}")\n\n\nif __name__ == "__main__":\n    main()