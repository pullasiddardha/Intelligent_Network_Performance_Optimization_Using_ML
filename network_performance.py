"""
Intelligent Network Performance Optimization Using ML
=====================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from collections import deque
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

class NetworkConfig:
    """Network simulation configuration parameters"""
    
    # Simulation parameters
    SIMULATION_TIME = 1000  # Time units
    TIME_SLOT = 0.1  # Time slot duration
    
    # Network capacity
    TOTAL_BANDWIDTH = 100  # Mbps
    BUFFER_SIZE = 100  # packets (increased for more realistic simulation)
    
    # Application characteristics
    VOIP_PACKET_SIZE = 160  # bytes (20ms of G.711)
    HTTP_PACKET_SIZE = 1500  # bytes (typical MTU)
    FTP_PACKET_SIZE = 1500  # bytes
    
    # Traffic arrival rates (packets per time unit) - Poisson lambda
    VOIP_ARRIVAL_RATE = 2.0
    HTTP_ARRIVAL_RATE = 2.5
    FTP_ARRIVAL_RATE = 3.5
    
    # Service rates (packets per time unit)
    SERVICE_RATE = 15  # Base service rate
    
    # Priority levels for Static QoS
    PRIORITY_VOIP = 3  # Highest
    PRIORITY_HTTP = 2
    PRIORITY_FTP = 1   # Lowest
    
    # ML Configuration
    ML_TRAIN_SIZE = 0.8
    ML_RANDOM_STATE = 42
    ML_CV_FOLDS = 5
    
    # Congestion thresholds for classification
    CONGESTION_LOW_THRESHOLD = 0.3
    CONGESTION_HIGH_THRESHOLD = 0.7


# ============================================================================
# PACKET CLASS
# ============================================================================

class Packet:
    """Represents a network packet"""
    
    packet_id_counter = 0
    
    def __init__(self, app_type, arrival_time, size):
        """
        Initialize a packet
        
        Args:
            app_type: Application type ('VoIP', 'HTTP', 'FTP')
            arrival_time: Time when packet arrived
            size: Packet size in bytes
        """
        self.packet_id = Packet.packet_id_counter
        Packet.packet_id_counter += 1
        self.app_type = app_type
        self.arrival_time = arrival_time
        self.size = size
        self.departure_time = None
        self.delay = None
        
    def set_departure(self, departure_time):
        """Set departure time and calculate delay"""
        self.departure_time = departure_time
        self.delay = departure_time - self.arrival_time


# ============================================================================
# TRAFFIC GENERATOR
# ============================================================================

class TrafficGenerator:
    """Generates Poisson traffic for different applications"""
    
    def __init__(self, config):
        self.config = config
        
    def generate_arrivals(self, app_type, arrival_rate, simulation_time):
        """
        Generate Poisson arrival times for an application
        
        Args:
            app_type: Application type
            arrival_rate: Lambda parameter for Poisson process
            simulation_time: Total simulation time
            
        Returns:
            List of (arrival_time, packet) tuples
        """
        packets = []
        current_time = 0
        
        # Determine packet size based on application type
        if app_type == 'VoIP':
            packet_size = self.config.VOIP_PACKET_SIZE
        elif app_type == 'HTTP':
            packet_size = self.config.HTTP_PACKET_SIZE
        else:  # FTP
            packet_size = self.config.FTP_PACKET_SIZE
        
        # Generate arrivals using Poisson process
        while current_time < simulation_time:
            # Inter-arrival time is exponentially distributed
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            
            if current_time < simulation_time:
                packet = Packet(app_type, current_time, packet_size)
                packets.append((current_time, packet))
        
        return packets
    
    def generate_all_traffic(self):
        """Generate traffic for all applications"""
        voip_traffic = self.generate_arrivals(
            'VoIP', 
            self.config.VOIP_ARRIVAL_RATE, 
            self.config.SIMULATION_TIME
        )
        http_traffic = self.generate_arrivals(
            'HTTP', 
            self.config.HTTP_ARRIVAL_RATE, 
            self.config.SIMULATION_TIME
        )
        ftp_traffic = self.generate_arrivals(
            'FTP', 
            self.config.FTP_ARRIVAL_RATE, 
            self.config.SIMULATION_TIME
        )
        
        # Merge and sort all traffic by arrival time
        all_traffic = voip_traffic + http_traffic + ftp_traffic
        all_traffic.sort(key=lambda x: x[0])
        
        return all_traffic


# ============================================================================
# NETWORK SIMULATORS
# ============================================================================

class FIFOSimulator:
    """Simulate network without QoS (single FIFO queue)"""
    
    def __init__(self, config):
        self.config = config
        self.queue = deque()
        self.metrics = {
            'delays': [],
            'throughput': [],
            'queue_length': [],
            'packet_loss': 0,
            'packets_transmitted': 0,
            'total_packets': 0
        }
        
    def simulate(self, traffic):
        """
        Simulate FIFO queuing
        
        Args:
            traffic: List of (arrival_time, packet) tuples
        """
        current_time = 0
        traffic_index = 0
        next_service_time = 0
        
        print("Simulating FIFO (No QoS)...")
        
        while current_time < self.config.SIMULATION_TIME or len(self.queue) > 0:
            # Process arrivals
            while (traffic_index < len(traffic) and 
                   traffic[traffic_index][0] <= current_time):
                arrival_time, packet = traffic[traffic_index]
                self.metrics['total_packets'] += 1
                
                # Check buffer overflow
                if len(self.queue) < self.config.BUFFER_SIZE:
                    self.queue.append(packet)
                else:
                    self.metrics['packet_loss'] += 1
                
                traffic_index += 1
            
            # Service packets
            if len(self.queue) > 0 and current_time >= next_service_time:
                packet = self.queue.popleft()
                
                # Calculate service time based on packet size and bandwidth
                service_time = (packet.size * 8) / (self.config.TOTAL_BANDWIDTH * 1e6) * 1e3
                service_time = max(service_time, self.config.TIME_SLOT)
                
                packet.set_departure(current_time + service_time)
                self.metrics['delays'].append(packet.delay)
                self.metrics['packets_transmitted'] += 1
                
                next_service_time = current_time + service_time
            
            # Record metrics
            self.metrics['queue_length'].append(len(self.queue))
            
            # Calculate throughput (packets per time unit)
            if len(self.metrics['throughput']) == 0:
                self.metrics['throughput'].append(0)
            else:
                # Moving average throughput
                window = min(10, len(self.metrics['delays']))
                if window > 0:
                    recent_packets = window
                    self.metrics['throughput'].append(recent_packets / (current_time + 1))
                else:
                    self.metrics['throughput'].append(0)
            
            current_time += self.config.TIME_SLOT
        
        return self.get_summary()
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'avg_delay': np.mean(self.metrics['delays']) if self.metrics['delays'] else 0,
            'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0,
            'packet_loss_rate': self.metrics['packet_loss'] / max(self.metrics['total_packets'], 1),
            'metrics': self.metrics
        }


class StaticPriorityQoSSimulator:
    """Simulate network with static priority QoS"""
    
    def __init__(self, config):
        self.config = config
        # Separate queues for each priority level
        self.queues = {
            'VoIP': deque(),
            'HTTP': deque(),
            'FTP': deque()
        }
        self.metrics = {
            'delays': [],
            'throughput': [],
            'queue_length': [],
            'packet_loss': 0,
            'packets_transmitted': 0,
            'total_packets': 0
        }
        
    def simulate(self, traffic):
        """
        Simulate static priority QoS
        
        Args:
            traffic: List of (arrival_time, packet) tuples
        """
        current_time = 0
        traffic_index = 0
        next_service_time = 0
        
        print("Simulating Static Priority QoS...")
        
        while current_time < self.config.SIMULATION_TIME or self._has_packets():
            # Process arrivals
            while (traffic_index < len(traffic) and 
                   traffic[traffic_index][0] <= current_time):
                arrival_time, packet = traffic[traffic_index]
                self.metrics['total_packets'] += 1
                
                # Check buffer overflow (per-queue)
                total_queue_size = sum(len(q) for q in self.queues.values())
                if total_queue_size < self.config.BUFFER_SIZE:
                    self.queues[packet.app_type].append(packet)
                else:
                    self.metrics['packet_loss'] += 1
                
                traffic_index += 1
            
            # Service packets (strict priority)
            if current_time >= next_service_time:
                packet = self._get_highest_priority_packet()
                
                if packet:
                    service_time = (packet.size * 8) / (self.config.TOTAL_BANDWIDTH * 1e6) * 1e3
                    service_time = max(service_time, self.config.TIME_SLOT)
                    
                    packet.set_departure(current_time + service_time)
                    self.metrics['delays'].append(packet.delay)
                    self.metrics['packets_transmitted'] += 1
                    
                    next_service_time = current_time + service_time
            
            # Record metrics
            total_queue = sum(len(q) for q in self.queues.values())
            self.metrics['queue_length'].append(total_queue)
            
            # Calculate throughput
            if len(self.metrics['throughput']) == 0:
                self.metrics['throughput'].append(0)
            else:
                window = min(10, len(self.metrics['delays']))
                if window > 0:
                    self.metrics['throughput'].append(window / (current_time + 1))
                else:
                    self.metrics['throughput'].append(0)
            
            current_time += self.config.TIME_SLOT
        
        return self.get_summary()
    
    def _get_highest_priority_packet(self):
        """Get packet from highest priority non-empty queue"""
        # Priority order: VoIP > HTTP > FTP
        for app_type in ['VoIP', 'HTTP', 'FTP']:
            if len(self.queues[app_type]) > 0:
                return self.queues[app_type].popleft()
        return None
    
    def _has_packets(self):
        """Check if any queue has packets"""
        return any(len(q) > 0 for q in self.queues.values())
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'avg_delay': np.mean(self.metrics['delays']) if self.metrics['delays'] else 0,
            'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0,
            'packet_loss_rate': self.metrics['packet_loss'] / max(self.metrics['total_packets'], 1),
            'metrics': self.metrics
        }


# ============================================================================
# MACHINE LEARNING MODULE
# ============================================================================

class CongestionPredictor:
    """ML-based congestion prediction system"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.feature_scaler = None
        
    def prepare_training_data(self, simulation_metrics):
        """
        Prepare training data from simulation metrics
        
        Features:
        - Arrival rate (packets in current window)
        - Queue length
        - Previous throughput
        
        Target:
        - Congestion level (0=Low, 1=Medium, 2=High)
        
        Args:
            simulation_metrics: Metrics from simulation
            
        Returns:
            X, y: Feature matrix and target vector
        """
        queue_lengths = simulation_metrics['queue_length']
        throughputs = simulation_metrics['throughput']
        
        features = []
        targets = []
        
        # Window size for calculating arrival rate
        window_size = 10
        
        for i in range(window_size, len(queue_lengths) - 1):
            # Feature 1: Arrival rate (estimated from queue length change)
            arrival_rate = max(0, queue_lengths[i] - queue_lengths[i-1] + 
                             (throughputs[i] if i < len(throughputs) else 0))
            
            # Feature 2: Current queue length
            queue_length = queue_lengths[i]
            
            # Feature 3: Previous throughput (moving average)
            prev_throughput = np.mean(throughputs[max(0, i-window_size):i]) if i > 0 else 0
            
            features.append([arrival_rate, queue_length, prev_throughput])
            
            # Target: Congestion at t+1
            # Based on queue utilization
            queue_utilization = queue_lengths[i+1] / self.config.BUFFER_SIZE
            
            if queue_utilization < self.config.CONGESTION_LOW_THRESHOLD:
                congestion_class = 0  # Low
            elif queue_utilization < self.config.CONGESTION_HIGH_THRESHOLD:
                congestion_class = 1  # Medium
            else:
                congestion_class = 2  # High
            
            targets.append(congestion_class)
        
        X = np.array(features)
        y = np.array(targets)
        
        return X, y
    
    def train_models(self, X, y):
        """
        Train and compare multiple ML models
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print("\n" + "="*70)
        print("MACHINE LEARNING MODEL TRAINING")
        print("="*70)
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=self.config.ML_TRAIN_SIZE, 
            random_state=self.config.ML_RANDOM_STATE,
            stratify=y
        )
        
        print(f"\nDataset Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Class distribution (train): {np.bincount(y_train)}")
        print(f"  Class distribution (test): {np.bincount(y_test)}")
        
        # Model 1: Decision Tree with hyperparameter tuning
        print("\n" + "-"*70)
        print("Training Decision Tree Classifier...")
        print("-"*70)
        
        dt_params = {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        dt_base = DecisionTreeClassifier(random_state=self.config.ML_RANDOM_STATE)
        dt_grid = GridSearchCV(
            dt_base, 
            dt_params, 
            cv=self.config.ML_CV_FOLDS,
            scoring='accuracy',
            n_jobs=-1
        )
        dt_grid.fit(X_train, y_train)
        
        best_dt = dt_grid.best_estimator_
        print(f"Best parameters: {dt_grid.best_params_}")
        
        # Cross-validation scores
        cv_scores_dt = cross_val_score(
            best_dt, X_train, y_train, 
            cv=self.config.ML_CV_FOLDS, 
            scoring='accuracy'
        )
        print(f"Cross-validation accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")
        
        # Test set evaluation
        y_pred_dt = best_dt.predict(X_test)
        self._evaluate_model("Decision Tree", y_test, y_pred_dt)
        
        self.models['Decision Tree'] = best_dt
        
        # Model 2: Random Forest with hyperparameter tuning
        print("\n" + "-"*70)
        print("Training Random Forest Classifier...")
        print("-"*70)
        
        rf_params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=self.config.ML_RANDOM_STATE)
        rf_grid = GridSearchCV(
            rf_base, 
            rf_params, 
            cv=self.config.ML_CV_FOLDS,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        best_rf = rf_grid.best_estimator_
        print(f"Best parameters: {rf_grid.best_params_}")
        
        # Cross-validation scores
        cv_scores_rf = cross_val_score(
            best_rf, X_train, y_train, 
            cv=self.config.ML_CV_FOLDS, 
            scoring='accuracy'
        )
        print(f"Cross-validation accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
        
        # Test set evaluation
        y_pred_rf = best_rf.predict(X_test)
        self._evaluate_model("Random Forest", y_test, y_pred_rf)
        
        self.models['Random Forest'] = best_rf
        
        # Select best model based on test accuracy
        dt_accuracy = accuracy_score(y_test, y_pred_dt)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        
        if rf_accuracy >= dt_accuracy:
            self.best_model = best_rf
            best_model_name = "Random Forest"
        else:
            self.best_model = best_dt
            best_model_name = "Decision Tree"
        
        print("\n" + "="*70)
        print(f"BEST MODEL SELECTED: {best_model_name}")
        print(f"Test Accuracy: {max(dt_accuracy, rf_accuracy):.4f}")
        print("="*70)
        
        return X_train, X_test, y_train, y_test
    
    def _evaluate_model(self, model_name, y_test, y_pred):
        """Evaluate and print model performance metrics"""
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{model_name} Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Low', 'Medium', 'High'],
                                   zero_division=0))
    
    def predict_congestion(self, arrival_rate, queue_length, prev_throughput):
        """
        Predict congestion level
        
        Args:
            arrival_rate: Current arrival rate
            queue_length: Current queue length
            prev_throughput: Previous throughput
            
        Returns:
            Congestion level (0=Low, 1=Medium, 2=High)
        """
        if self.best_model is None:
            return 1  # Default to medium if model not trained
        
        features = np.array([[arrival_rate, queue_length, prev_throughput]])
        prediction = self.best_model.predict(features)[0]
        
        return prediction


# ============================================================================
# ML-ADAPTIVE QoS SIMULATOR
# ============================================================================

class MLAdaptiveQoSSimulator:
    """Simulate network with ML-driven adaptive QoS"""
    
    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor
        
        # Separate queues for each application
        self.queues = {
            'VoIP': deque(),
            'HTTP': deque(),
            'FTP': deque()
        }
        
        self.metrics = {
            'delays': [],
            'throughput': [],
            'queue_length': [],
            'packet_loss': 0,
            'packets_transmitted': 0,
            'total_packets': 0,
            'congestion_predictions': []
        }
        
    def simulate(self, traffic):
        """
        Simulate ML-adaptive QoS
        
        Args:
            traffic: List of (arrival_time, packet) tuples
        """
        current_time = 0
        traffic_index = 0
        next_service_time = 0
        
        # For calculating features
        arrival_window = deque(maxlen=10)
        throughput_window = deque(maxlen=10)
        
        print("Simulating ML-Adaptive QoS...")
        
        while current_time < self.config.SIMULATION_TIME or self._has_packets():
            # Process arrivals
            arrivals_in_slot = 0
            while (traffic_index < len(traffic) and 
                   traffic[traffic_index][0] <= current_time):
                arrival_time, packet = traffic[traffic_index]
                self.metrics['total_packets'] += 1
                arrivals_in_slot += 1
                
                # Check buffer overflow
                total_queue_size = sum(len(q) for q in self.queues.values())
                if total_queue_size < self.config.BUFFER_SIZE:
                    self.queues[packet.app_type].append(packet)
                else:
                    # Drop packet - buffer overflow
                    self.metrics['packet_loss'] += 1
                
                traffic_index += 1
            
            arrival_window.append(arrivals_in_slot)
            
            # Calculate features for ML prediction
            total_queue = sum(len(q) for q in self.queues.values())
            arrival_rate = np.mean(arrival_window) if len(arrival_window) > 0 else 0
            prev_throughput = np.mean(throughput_window) if len(throughput_window) > 0 else 0
            
            # ML prediction of congestion
            congestion_level = self.predictor.predict_congestion(
                arrival_rate, total_queue, prev_throughput
            )
            self.metrics['congestion_predictions'].append(congestion_level)
            
            # Get dynamic priorities based on ML prediction
            priorities = self._get_dynamic_priorities(congestion_level)
            
            # Service packet using adaptive priority
            packets_served = 0
            if current_time >= next_service_time:
                packet = self._get_packet_by_priority(priorities)
                
                if packet:
                    # Service time based on full bandwidth (like FIFO and Static)
                    service_time = (packet.size * 8) / (self.config.TOTAL_BANDWIDTH * 1e6) * 1e3
                    service_time = max(service_time, self.config.TIME_SLOT)
                    
                    packet.set_departure(current_time + service_time)
                    self.metrics['delays'].append(packet.delay)
                    self.metrics['packets_transmitted'] += 1
                    packets_served = 1
                    
                    next_service_time = current_time + service_time
            
            throughput_window.append(packets_served)
            
            # Record metrics
            self.metrics['queue_length'].append(total_queue)
            
            # Calculate throughput
            if len(throughput_window) > 0:
                self.metrics['throughput'].append(np.mean(throughput_window))
            else:
                self.metrics['throughput'].append(0)
            
            current_time += self.config.TIME_SLOT
        
        return self.get_summary()
    
    def _get_dynamic_priorities(self, congestion_level):
        """
        Get dynamic priority weights based on congestion level
        
        Higher number = higher priority
        
        Args:
            congestion_level: 0 (Low), 1 (Medium), 2 (High)
            
        Returns:
            Dictionary of priority weights
        """
        if congestion_level == 0:  # Low congestion
            # More balanced priorities
            return {'VoIP': 5, 'HTTP': 4, 'FTP': 3}
        elif congestion_level == 1:  # Medium congestion
            # Moderate VoIP preference
            return {'VoIP': 6, 'HTTP': 3, 'FTP': 2}
        else:  # High congestion
            # Strong VoIP protection with minimum service for others
            return {'VoIP': 8, 'HTTP': 2, 'FTP': 1}
    
    def _get_packet_by_priority(self, priorities):
        """
        Select next packet using weighted priority
        
        Uses probabilistic selection based on queue length and priority
        to avoid starvation while maintaining priority ordering
        
        Args:
            priorities: Dictionary of priority weights
            
        Returns:
            Next packet to serve, or None
        """
        # Build weighted list of candidate queues
        candidates = []
        weights = []
        
        for app_type in ['VoIP', 'HTTP', 'FTP']:
            queue_len = len(self.queues[app_type])
            if queue_len > 0:
                # Weight = priority * queue_pressure
                # Higher priority and longer queues get more weight
                weight = priorities[app_type] * (1 + np.log1p(queue_len))
                candidates.append(app_type)
                weights.append(weight)
        
        if not candidates:
            return None
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Select queue probabilistically
        selected_app = np.random.choice(candidates, p=weights)
        
        return self.queues[selected_app].popleft()
    
    def _has_packets(self):
        """Check if any queue has packets"""
        return any(len(q) > 0 for q in self.queues.values())
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'avg_delay': np.mean(self.metrics['delays']) if self.metrics['delays'] else 0,
            'avg_throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0,
            'packet_loss_rate': self.metrics['packet_loss'] / max(self.metrics['total_packets'], 1),
            'metrics': self.metrics
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison_results(fifo_metrics, static_metrics, ml_metrics, config):
    """
    Create comprehensive comparison plots
    
    Args:
        fifo_metrics: Metrics from FIFO simulation
        static_metrics: Metrics from Static QoS simulation
        ml_metrics: Metrics from ML-Adaptive QoS simulation
        config: Network configuration
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Network QoS Performance Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting (downsample for clarity)
    sample_rate = 10
    
    # Find minimum length across all metrics
    min_length = min(
        len(fifo_metrics['queue_length']),
        len(static_metrics['queue_length']),
        len(ml_metrics['queue_length'])
    )
    
    time_points = np.arange(0, min_length, sample_rate) * config.TIME_SLOT
    
    # Plot 1: Queue Length Comparison
    ax1 = axes[0, 0]
    
    # Calculate rolling average queue lengths
    window = 50
    fifo_queue_smooth = pd.Series(fifo_metrics['queue_length'][:min_length]).rolling(window, min_periods=1).mean()
    static_queue_smooth = pd.Series(static_metrics['queue_length'][:min_length]).rolling(window, min_periods=1).mean()
    ml_queue_smooth = pd.Series(ml_metrics['queue_length'][:min_length]).rolling(window, min_periods=1).mean()
    
    # Ensure consistent lengths after sampling
    n_points = len(time_points)
    ax1.plot(time_points, fifo_queue_smooth[::sample_rate][:n_points], label='FIFO (No QoS)', 
             linewidth=2, alpha=0.8, color='red')
    ax1.plot(time_points, static_queue_smooth[::sample_rate][:n_points], label='Static Priority QoS', 
             linewidth=2, alpha=0.8, color='blue')
    ax1.plot(time_points, ml_queue_smooth[::sample_rate][:n_points], label='ML-Adaptive QoS', 
             linewidth=2, alpha=0.8, color='green')
    
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Queue Length (packets)', fontsize=11)
    ax1.set_title('Queue Length Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Throughput Comparison
    ax2 = axes[0, 1]
    
    min_throughput_length = min(
        len(fifo_metrics['throughput']),
        len(static_metrics['throughput']),
        len(ml_metrics['throughput']),
        min_length
    )
    
    fifo_throughput = pd.Series(fifo_metrics['throughput'][:min_throughput_length]).rolling(window, min_periods=1).mean()
    static_throughput = pd.Series(static_metrics['throughput'][:min_throughput_length]).rolling(window, min_periods=1).mean()
    ml_throughput = pd.Series(ml_metrics['throughput'][:min_throughput_length]).rolling(window, min_periods=1).mean()
    
    time_points_tp = np.arange(0, min_throughput_length, sample_rate) * config.TIME_SLOT
    n_points_tp = len(time_points_tp)
    
    ax2.plot(time_points_tp, fifo_throughput[::sample_rate][:n_points_tp], label='FIFO (No QoS)', 
             linewidth=2, alpha=0.8, color='red')
    ax2.plot(time_points_tp, static_throughput[::sample_rate][:n_points_tp], label='Static Priority QoS', 
             linewidth=2, alpha=0.8, color='blue')
    ax2.plot(time_points_tp, ml_throughput[::sample_rate][:n_points_tp], label='ML-Adaptive QoS', 
             linewidth=2, alpha=0.8, color='green')
    
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Throughput (packets/time)', fontsize=11)
    ax2.set_title('Throughput Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Packet Loss Comparison (Bar Chart)
    ax3 = axes[1, 0]
    
    schemes = ['FIFO\n(No QoS)', 'Static\nPriority QoS', 'ML-Adaptive\nQoS']
    loss_rates = [
        fifo_metrics['packet_loss'] / max(fifo_metrics['total_packets'], 1) * 100,
        static_metrics['packet_loss'] / max(static_metrics['total_packets'], 1) * 100,
        ml_metrics['packet_loss'] / max(ml_metrics['total_packets'], 1) * 100
    ]
    
    colors = ['red', 'blue', 'green']
    bars = ax3.bar(schemes, loss_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, loss_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Packet Loss Rate (%)', fontsize=11)
    ax3.set_title('Packet Loss Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Delay Distribution (Box Plot)
    ax4 = axes[1, 1]
    
    # Sample delays for visualization
    delay_data = [
        fifo_metrics['delays'][::5],
        static_metrics['delays'][::5],
        ml_metrics['delays'][::5]
    ]
    
    bp = ax4.boxplot(delay_data, labels=schemes, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_ylabel('Delay (time units)', fontsize=11)
    ax4.set_title('Delay Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    os.makedirs('outputs', exist_ok=True)

    plt.savefig('outputs/qos_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Comparison plots saved to: qos_comparison.png")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*70)
    print(" INTELLIGENT NETWORK QoS OPTIMIZATION USING MACHINE LEARNING")
    print("="*70)
    
    # Initialize configuration
    config = NetworkConfig()
    
    # Generate traffic
    print("\nGenerating Poisson traffic...")
    traffic_gen = TrafficGenerator(config)
    all_traffic = traffic_gen.generate_all_traffic()
    print(f"✓ Generated {len(all_traffic)} packets")
    
    # ========================================================================
    # SCENARIO 1: FIFO (No QoS)
    # ========================================================================
    print("\n" + "="*70)
    print("SCENARIO 1: FIFO Queue (No QoS)")
    print("="*70)
    
    fifo_sim = FIFOSimulator(config)
    fifo_results = fifo_sim.simulate(all_traffic)
    
    # ========================================================================
    # SCENARIO 2: Static Priority QoS
    # ========================================================================
    print("\n" + "="*70)
    print("SCENARIO 2: Static Priority QoS")
    print("="*70)
    
    static_sim = StaticPriorityQoSSimulator(config)
    static_results = static_sim.simulate(all_traffic)
    
    # ========================================================================
    # MACHINE LEARNING: Train Congestion Predictor
    # ========================================================================
    
    # Use FIFO simulation data for training (realistic network behavior)
    predictor = CongestionPredictor(config)
    X, y = predictor.prepare_training_data(fifo_results['metrics'])
    
    print(f"\n✓ Prepared {len(X)} training samples")
    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution: {np.bincount(y)}")
    
    # Train models with cross-validation and hyperparameter tuning
    X_train, X_test, y_train, y_test = predictor.train_models(X, y)
    
    # ========================================================================
    # SCENARIO 3: ML-Adaptive QoS
    # ========================================================================
    print("\n" + "="*70)
    print("SCENARIO 3: ML-Adaptive QoS")
    print("="*70)
    
    ml_sim = MLAdaptiveQoSSimulator(config, predictor)
    ml_results = ml_sim.simulate(all_traffic)
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    summary_data = {
        'Metric': ['Average Delay', 'Average Throughput', 'Packet Loss Rate (%)'],
        'FIFO (No QoS)': [
            f"{fifo_results['avg_delay']:.4f}",
            f"{fifo_results['avg_throughput']:.4f}",
            f"{fifo_results['packet_loss_rate']*100:.2f}"
        ],
        'Static Priority QoS': [
            f"{static_results['avg_delay']:.4f}",
            f"{static_results['avg_throughput']:.4f}",
            f"{static_results['packet_loss_rate']*100:.2f}"
        ],
        'ML-Adaptive QoS': [
            f"{ml_results['avg_delay']:.4f}",
            f"{ml_results['avg_throughput']:.4f}",
            f"{ml_results['packet_loss_rate']*100:.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "-"*70)
    print("IMPROVEMENT ANALYSIS (ML-Adaptive vs FIFO)")
    print("-"*70)
    
    delay_improvement = ((fifo_results['avg_delay'] - ml_results['avg_delay']) / 
                        fifo_results['avg_delay'] * 100)
    throughput_improvement = ((ml_results['avg_throughput'] - fifo_results['avg_throughput']) / 
                             fifo_results['avg_throughput'] * 100)
    loss_improvement = ((fifo_results['packet_loss_rate'] - ml_results['packet_loss_rate']) / 
                       fifo_results['packet_loss_rate'] * 100)
    
    print(f"Delay Reduction: {delay_improvement:.2f}%")
    print(f"Throughput Improvement: {throughput_improvement:.2f}%")
    print(f"Packet Loss Reduction: {loss_improvement:.2f}%")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    
    plot_comparison_results(
        fifo_results['metrics'],
        static_results['metrics'],
        ml_results['metrics'],
        config
    )
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print("\n✓ All results generated successfully!")
    print("✓ Machine learning models trained and validated")
    print("✓ Performance comparison completed")
    print("✓ Visualization saved")
    
    return {
        'fifo': fifo_results,
        'static': static_results,
        'ml_adaptive': ml_results,
        'predictor': predictor
    }


if __name__ == "__main__":
    results = main()