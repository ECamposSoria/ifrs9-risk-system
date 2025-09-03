#!/usr/bin/env python3
"""
IFRS9 Polars Performance Metrics Exporter
Collects and exposes Polars-specific performance metrics for Prometheus monitoring
"""

import os
import sys
import time
import psutil
import polars as pl
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily
import threading
import logging
from datetime import datetime, timedelta
import json
import gc
import tracemalloc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PolarsQueryMetrics:
    """Container for Polars query performance metrics"""
    query_id: str
    start_time: float
    end_time: float
    duration: float
    memory_used: int
    memory_peak: int
    rows_processed: int
    columns_processed: int
    operation_type: str
    optimization_applied: bool
    lazy_execution: bool
    streaming_mode: bool
    error_occurred: bool = False
    error_message: Optional[str] = None

class PolarsPerformanceCollector:
    """Custom Prometheus collector for Polars performance metrics"""
    
    def __init__(self):
        self.query_metrics: List[PolarsQueryMetrics] = []
        self.metrics_lock = threading.Lock()
        
        # Initialize Prometheus metrics
        self.registry = CollectorRegistry()
        
        # Query performance metrics
        self.query_duration_seconds = Histogram(
            'ifrs9_polars_query_duration_seconds',
            'Duration of Polars queries in seconds',
            ['operation_type', 'lazy_execution', 'streaming_mode', 'optimization_applied'],
            registry=self.registry
        )
        
        self.query_total = Counter(
            'ifrs9_polars_queries_total',
            'Total number of Polars queries executed',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'ifrs9_polars_memory_usage_bytes',
            'Current Polars memory usage in bytes',
            registry=self.registry
        )
        
        self.memory_peak_bytes = Histogram(
            'ifrs9_polars_memory_peak_bytes',
            'Peak memory usage during Polars operations',
            ['operation_type'],
            registry=self.registry
        )
        
        self.rows_processed_total = Counter(
            'ifrs9_polars_rows_processed_total',
            'Total number of rows processed by Polars',
            ['operation_type'],
            registry=self.registry
        )
        
        self.columns_processed_total = Counter(
            'ifrs9_polars_columns_processed_total',
            'Total number of columns processed by Polars',
            ['operation_type'],
            registry=self.registry
        )
        
        # Performance efficiency metrics
        self.throughput_rows_per_second = Gauge(
            'ifrs9_polars_throughput_rows_per_second',
            'Polars processing throughput in rows per second',
            ['operation_type'],
            registry=self.registry
        )
        
        self.memory_efficiency = Gauge(
            'ifrs9_polars_memory_efficiency',
            'Rows processed per MB of memory used',
            ['operation_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.query_errors_total = Counter(
            'ifrs9_polars_query_errors_total',
            'Total number of Polars query errors',
            ['operation_type', 'error_type'],
            registry=self.registry
        )
        
        # Resource utilization
        self.cpu_usage_percent = Gauge(
            'ifrs9_polars_cpu_usage_percent',
            'CPU usage percentage during Polars operations',
            registry=self.registry
        )
        
        # Lazy evaluation benefits
        self.lazy_optimization_savings = Gauge(
            'ifrs9_polars_lazy_optimization_savings_percent',
            'Performance improvement from lazy evaluation optimizations',
            ['optimization_type'],
            registry=self.registry
        )
        
        # DataFrame size metrics
        self.dataframe_size_bytes = Histogram(
            'ifrs9_polars_dataframe_size_bytes',
            'Size of Polars DataFrames in bytes',
            ['data_type'],
            registry=self.registry
        )
        
        # Start background monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring threads"""
        # Resource monitoring thread
        resource_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        resource_thread.start()
        
        # Metrics cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        cleanup_thread.start()
        
        logger.info("Polars performance monitoring started")
    
    def _monitor_resources(self):
        """Monitor system resources during Polars operations"""
        while True:
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_percent.set(cpu_percent)
                
                # Monitor memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                self.memory_usage_bytes.set(memory_info.rss)
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(30)
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks"""
        while True:
            try:
                cutoff_time = time.time() - 3600  # Keep metrics for 1 hour
                
                with self.metrics_lock:
                    self.query_metrics = [
                        metric for metric in self.query_metrics
                        if metric.end_time > cutoff_time
                    ]
                
                # Force garbage collection
                gc.collect()
                
                time.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
                time.sleep(600)
    
    def record_query_start(self, query_id: str, operation_type: str, 
                          lazy_execution: bool = False, streaming_mode: bool = False) -> str:
        """Record the start of a Polars query"""
        try:
            # Start memory tracing
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            
            start_time = time.time()
            
            # Store start context
            context = {
                'query_id': query_id,
                'start_time': start_time,
                'operation_type': operation_type,
                'lazy_execution': lazy_execution,
                'streaming_mode': streaming_mode,
                'start_memory': tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
            }
            
            return query_id
            
        except Exception as e:
            logger.error(f"Error recording query start: {e}")
            return query_id
    
    def record_query_end(self, query_id: str, result_df: Optional[pl.DataFrame] = None,
                        error: Optional[Exception] = None, optimization_applied: bool = False):
        """Record the completion of a Polars query"""
        try:
            end_time = time.time()
            
            # Get memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
            
            # Extract DataFrame metrics if available
            rows_processed = result_df.height if result_df is not None else 0
            columns_processed = result_df.width if result_df is not None else 0
            
            # Determine operation type from query_id or context
            operation_type = self._extract_operation_type(query_id)
            
            # Create metrics record
            metrics = PolarsQueryMetrics(
                query_id=query_id,
                start_time=0,  # Will be updated from context if available
                end_time=end_time,
                duration=0,  # Will be calculated
                memory_used=current_memory,
                memory_peak=peak_memory,
                rows_processed=rows_processed,
                columns_processed=columns_processed,
                operation_type=operation_type,
                optimization_applied=optimization_applied,
                lazy_execution='lazy' in query_id.lower(),
                streaming_mode='stream' in query_id.lower(),
                error_occurred=error is not None,
                error_message=str(error) if error else None
            )
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Store metrics
            with self.metrics_lock:
                self.query_metrics.append(metrics)
            
            logger.debug(f"Recorded query completion: {query_id}")
            
        except Exception as e:
            logger.error(f"Error recording query end: {e}")
    
    def _extract_operation_type(self, query_id: str) -> str:
        """Extract operation type from query ID"""
        query_lower = query_id.lower()
        
        if 'select' in query_lower or 'filter' in query_lower:
            return 'select'
        elif 'join' in query_lower:
            return 'join'
        elif 'group' in query_lower or 'agg' in query_lower:
            return 'aggregation'
        elif 'sort' in query_lower:
            return 'sort'
        elif 'read' in query_lower or 'load' in query_lower:
            return 'read'
        elif 'write' in query_lower or 'save' in query_lower:
            return 'write'
        elif 'transform' in query_lower:
            return 'transform'
        else:
            return 'other'
    
    def _update_prometheus_metrics(self, metrics: PolarsQueryMetrics):
        """Update Prometheus metrics based on query metrics"""
        try:
            # Labels
            labels = {
                'operation_type': metrics.operation_type,
                'lazy_execution': str(metrics.lazy_execution),
                'streaming_mode': str(metrics.streaming_mode),
                'optimization_applied': str(metrics.optimization_applied)
            }
            
            # Duration metric
            self.query_duration_seconds.labels(**labels).observe(metrics.duration)
            
            # Query counter
            status = 'error' if metrics.error_occurred else 'success'
            self.query_total.labels(
                operation_type=metrics.operation_type,
                status=status
            ).inc()
            
            # Memory metrics
            if metrics.memory_peak > 0:
                self.memory_peak_bytes.labels(
                    operation_type=metrics.operation_type
                ).observe(metrics.memory_peak)
            
            # Processing metrics
            if metrics.rows_processed > 0:
                self.rows_processed_total.labels(
                    operation_type=metrics.operation_type
                ).inc(metrics.rows_processed)
                
                # Calculate throughput
                if metrics.duration > 0:
                    throughput = metrics.rows_processed / metrics.duration
                    self.throughput_rows_per_second.labels(
                        operation_type=metrics.operation_type
                    ).set(throughput)
                
                # Calculate memory efficiency
                if metrics.memory_peak > 0:
                    efficiency = metrics.rows_processed / (metrics.memory_peak / 1024 / 1024)  # rows per MB
                    self.memory_efficiency.labels(
                        operation_type=metrics.operation_type
                    ).set(efficiency)
            
            if metrics.columns_processed > 0:
                self.columns_processed_total.labels(
                    operation_type=metrics.operation_type
                ).inc(metrics.columns_processed)
            
            # Error metrics
            if metrics.error_occurred:
                error_type = self._classify_error(metrics.error_message)
                self.query_errors_total.labels(
                    operation_type=metrics.operation_type,
                    error_type=error_type
                ).inc()
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _classify_error(self, error_message: Optional[str]) -> str:
        """Classify error type from error message"""
        if not error_message:
            return 'unknown'
        
        error_lower = error_message.lower()
        
        if 'memory' in error_lower or 'oom' in error_lower:
            return 'memory'
        elif 'timeout' in error_lower:
            return 'timeout'
        elif 'schema' in error_lower:
            return 'schema'
        elif 'type' in error_lower:
            return 'type'
        elif 'io' in error_lower or 'file' in error_lower:
            return 'io'
        else:
            return 'other'
    
    def record_dataframe_size(self, df: pl.DataFrame, data_type: str):
        """Record DataFrame size metrics"""
        try:
            # Estimate DataFrame size
            size_bytes = df.estimated_size()
            
            self.dataframe_size_bytes.labels(data_type=data_type).observe(size_bytes)
            
            logger.debug(f"Recorded DataFrame size: {data_type} - {size_bytes} bytes")
            
        except Exception as e:
            logger.error(f"Error recording DataFrame size: {e}")
    
    def record_lazy_optimization(self, optimization_type: str, savings_percent: float):
        """Record lazy evaluation optimization benefits"""
        try:
            self.lazy_optimization_savings.labels(
                optimization_type=optimization_type
            ).set(savings_percent)
            
            logger.debug(f"Recorded lazy optimization: {optimization_type} - {savings_percent}% savings")
            
        except Exception as e:
            logger.error(f"Error recording lazy optimization: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        try:
            with self.metrics_lock:
                if not self.query_metrics:
                    return {}
                
                # Calculate summary statistics
                durations = [m.duration for m in self.query_metrics if m.duration > 0]
                memory_peaks = [m.memory_peak for m in self.query_metrics if m.memory_peak > 0]
                throughputs = [
                    m.rows_processed / m.duration 
                    for m in self.query_metrics 
                    if m.duration > 0 and m.rows_processed > 0
                ]
                
                summary = {
                    'total_queries': len(self.query_metrics),
                    'successful_queries': sum(1 for m in self.query_metrics if not m.error_occurred),
                    'failed_queries': sum(1 for m in self.query_metrics if m.error_occurred),
                    'avg_duration_seconds': sum(durations) / len(durations) if durations else 0,
                    'max_duration_seconds': max(durations) if durations else 0,
                    'avg_memory_peak_mb': sum(memory_peaks) / len(memory_peaks) / 1024 / 1024 if memory_peaks else 0,
                    'max_memory_peak_mb': max(memory_peaks) / 1024 / 1024 if memory_peaks else 0,
                    'avg_throughput_rows_per_sec': sum(throughputs) / len(throughputs) if throughputs else 0,
                    'total_rows_processed': sum(m.rows_processed for m in self.query_metrics),
                    'lazy_query_percentage': sum(1 for m in self.query_metrics if m.lazy_execution) / len(self.query_metrics) * 100,
                    'streaming_query_percentage': sum(1 for m in self.query_metrics if m.streaming_mode) / len(self.query_metrics) * 100
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}

# Context manager for automatic query tracking
class PolarsQueryTracker:
    """Context manager for tracking Polars query performance"""
    
    def __init__(self, collector: PolarsPerformanceCollector, query_id: str, 
                 operation_type: str, lazy_execution: bool = False, 
                 streaming_mode: bool = False):
        self.collector = collector
        self.query_id = query_id
        self.operation_type = operation_type
        self.lazy_execution = lazy_execution
        self.streaming_mode = streaming_mode
        self.result = None
        self.error = None
        self.optimization_applied = False
    
    def __enter__(self):
        self.collector.record_query_start(
            self.query_id, self.operation_type, 
            self.lazy_execution, self.streaming_mode
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.error = exc_val
        
        self.collector.record_query_end(
            self.query_id, self.result, self.error, self.optimization_applied
        )
        
        return False  # Don't suppress exceptions
    
    def set_result(self, result_df: pl.DataFrame):
        """Set the result DataFrame"""
        self.result = result_df
    
    def set_optimization_applied(self, applied: bool = True):
        """Mark that query optimization was applied"""
        self.optimization_applied = applied

# Global collector instance
polars_collector = PolarsPerformanceCollector()

def start_polars_metrics_server(port: int = 9092):
    """Start the Polars metrics HTTP server"""
    try:
        start_http_server(port, registry=polars_collector.registry)
        logger.info(f"Polars metrics server started on port {port}")
        
        # Keep the server running
        while True:
            time.sleep(60)
            
            # Log performance summary periodically
            summary = polars_collector.get_performance_summary()
            if summary:
                logger.info(f"Polars performance summary: {json.dumps(summary, indent=2)}")
                
    except Exception as e:
        logger.error(f"Error starting metrics server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Configuration
    port = int(os.getenv('POLARS_METRICS_PORT', 9092))
    
    # Start the metrics server
    logger.info("Starting IFRS9 Polars Performance Metrics Exporter")
    start_polars_metrics_server(port)