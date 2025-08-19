# translation_model.py
import torch
import torch.nn as nn
from transformers import (
    MarianMTModel, MarianTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    EarlyStoppingCallback, AutoConfig
)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import joblib
import logging
import os
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict
import psutil
import GPUtil
import traceback
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/translation_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

@dataclass
class TrainingConfig:
    """Production configuration for training"""
    csv_file: str
    output_dir: str = "./model_output"
    epochs: int = 10  # Increased epochs for better training
    batch_size: int = 16  # Increased batch size for efficiency
    max_length: int = 512
    learning_rate: float = 3e-5  # Reduced learning rate for fine-tuning
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_pin_memory: bool = False
    save_total_limit: int = 5  # Increased save limit
    eval_steps: int = 200  # More frequent evaluation
    save_steps: int = 200  # More frequent saving
    logging_steps: int = 50  # More frequent logging
    early_stopping_patience: int = 5  # Increased patience
    validation_split: float = 0.15  # Increased validation split
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_all_data: bool = True  # Use all data for training
    
    def _post_init_(self):
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("checkpoints").mkdir(exist_ok=True)

class ProductionTranslationDataset(Dataset):
    """Production-ready dataset with comprehensive error handling"""
    
    def _init_(self, csv_file: str, tokenizer_en_to_ps: MarianTokenizer, 
                 tokenizer_ps_to_en: MarianTokenizer, config: TrainingConfig):
        self.config = config
        self.tokenizer_en_to_ps = tokenizer_en_to_ps
        self.tokenizer_ps_to_en = tokenizer_ps_to_en
        
        # Load and validate data
        try:
            self.data = pd.read_csv(csv_file, encoding='utf-8')
            logger.info(f"Loaded {len(self.data)} samples from {csv_file}")
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            raise
            
        # Validate columns
        if len(self.data.columns) < 2:
            raise ValueError(f"CSV must have at least 2 columns, found {len(self.data.columns)}")
            
        self.ps_col = self.data.columns[0]
        self.en_col = self.data.columns[1]
        
        # Data validation and cleaning
        self._validate_and_clean_data()
        
    def _validate_and_clean_data(self):
        """Validate and clean dataset"""
        initial_length = len(self.data)
        
        # Remove rows with missing values
        self.data = self.data.dropna(subset=[self.ps_col, self.en_col])
        
        # Remove rows with empty strings
        self.data = self.data[
            (self.data[self.ps_col].str.len() > 0) & 
            (self.data[self.en_col].str.len() > 0)
        ]
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        # Filter out extremely long sequences
        self.data = self.data[
            (self.data[self.ps_col].str.len() <= 1000) & 
            (self.data[self.en_col].str.len() <= 1000)
        ]
        
        cleaned_length = len(self.data)
        logger.info(f"Data cleaning: {initial_length} -> {cleaned_length} samples")
        
        if cleaned_length == 0:
            raise ValueError("No valid data remaining after cleaning")
    
    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx: int) -> Dict:
        """Get item with comprehensive error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ps_text = str(self.data.iloc[idx][self.ps_col]).strip()
                en_text = str(self.data.iloc[idx][self.en_col]).strip()
                
                if not ps_text or not en_text:
                    raise ValueError(f"Empty text at index {idx}")
                
                # Tokenize both directions with error handling
                ps_tokens = self._safe_tokenize(
                    ps_text, self.tokenizer_en_to_ps, "Pashto"
                )
                en_tokens = self._safe_tokenize(
                    en_text, self.tokenizer_ps_to_en, "English"
                )
                
                return {
                    'input_ids_ps': ps_tokens['input_ids'].squeeze(),
                    'attention_mask_ps': ps_tokens['attention_mask'].squeeze(),
                    'labels_ps': en_tokens['input_ids'].squeeze(),
                    
                    'input_ids_en': en_tokens['input_ids'].squeeze(),
                    'attention_mask_en': en_tokens['attention_mask'].squeeze(),
                    'labels_en': ps_tokens['input_ids'].squeeze(),
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for index {idx}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries exceeded for index {idx}")
                    # Return dummy data to prevent crashes
                    return self._get_dummy_item()
                continue
                
    def _safe_tokenize(self, text: str, tokenizer: MarianTokenizer, lang: str) -> Dict:
        """Safe tokenization with error handling"""
        try:
            return tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Tokenization failed for {lang} text: {text[:50]}... Error: {str(e)}")
            raise
    
    def _get_dummy_item(self) -> Dict:
        """Return dummy item to prevent training crashes"""
        dummy_tensor = torch.zeros(self.config.max_length, dtype=torch.long)
        return {
            'input_ids_ps': dummy_tensor.clone(),
            'attention_mask_ps': dummy_tensor.clone(),
            'labels_ps': dummy_tensor.clone(),
            
            'input_ids_en': dummy_tensor.clone(),
            'attention_mask_en': dummy_tensor.clone(),
            'labels_en': dummy_tensor.clone(),
        }

class DualTranslationModel(nn.Module):
    """Enhanced dual translation model with bidirectional capabilities"""
    
    def _init_(self, model_en_to_ps: MarianMTModel, model_ps_to_en: MarianMTModel):
        super()._init_()
        self.model_en_to_ps = model_en_to_ps
        self.model_ps_to_en = model_ps_to_en
        
        # Partial fine-tuning for better performance
        self._partial_fine_tuning()
        
    def _partial_fine_tuning(self):
        """Partially unfreeze model layers for better fine-tuning"""
        # Freeze embeddings
        for param in self.model_en_to_ps.model.encoder.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model_en_to_ps.model.decoder.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model_ps_to_en.model.encoder.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model_ps_to_en.model.decoder.embed_tokens.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers
        for layer in self.model_en_to_ps.model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.model_en_to_ps.model.decoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.model_ps_to_en.model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.model_ps_to_en.model.decoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, labels, direction='en_to_ps'):
        try:
            if direction == 'en_to_ps':
                outputs = self.model_en_to_ps(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            else:
                outputs = self.model_ps_to_en(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            return outputs
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

class SystemMonitor:
    """Monitor system resources during training"""
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            gpu_stats = GPUtil.getGPUs()
            
            stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
            }
            
            if gpu_stats:
                stats['gpu_utilization'] = gpu_stats[0].load * 100
                stats['gpu_memory_percent'] = gpu_stats[0].memoryUtil * 100
            
            return stats
        except Exception as e:
            logger.warning(f"Failed to get system stats: {str(e)}")
            return {}

class ProductionTrainerCallback:
    """Custom callback for production training with comprehensive logging"""
    
    def _init_(self, config: TrainingConfig):
        self.config = config
        self.system_monitor = SystemMonitor()
        self.best_eval_loss = float('inf')
        self.best_model_state = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Custom logging with system metrics and best model tracking"""
        if logs:
            # Add system metrics
            system_stats = self.system_monitor.get_system_stats()
            logs.update(system_stats)
            
            # Track best model
            if 'eval_loss' in logs:
                current_eval_loss = logs['eval_loss']
                if current_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = current_eval_loss
                    logger.info(f"New best evaluation loss: {current_eval_loss:.6f}")
            
            # Log training progress
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                logger.info(f"Evaluation Loss = {logs['eval_loss']:.4f}")
            if 'learning_rate' in logs:
                logger.info(f"Learning Rate = {logs['learning_rate']:.6f}")
            
            # Log system stats
            if system_stats:
                logger.info(f"System Stats: {system_stats}")

def create_translation_model(config: TrainingConfig) -> Tuple[DualTranslationModel, MarianTokenizer, MarianTokenizer]:
    """Initialize production-ready translation models"""
    try:
        logger.info("Initializing pre-trained translation models...")
        
        # Load models with error handling
        model_en_to_ps = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-ps",
            torch_dtype=torch.float16 if config.fp16 and config.device == "cuda" else torch.float32
        )
        tokenizer_en_to_ps = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ps")
        
        model_ps_to_en = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-ps-en",
            torch_dtype=torch.float16 if config.fp16 and config.device == "cuda" else torch.float32
        )
        tokenizer_ps_to_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ps-en")
        
        # Move to device
        if config.device == "cuda":
            model_en_to_ps = model_en_to_ps.cuda()
            model_ps_to_en = model_ps_to_en.cuda()
        
        # Create combined model
        model = DualTranslationModel(model_en_to_ps, model_ps_to_en)
        
        logger.info("Successfully loaded pre-trained translation models")
        return model, tokenizer_en_to_ps, tokenizer_ps_to_en
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model(config: TrainingConfig) -> Tuple[DualTranslationModel, MarianTokenizer, MarianTokenizer]:
    """Production training with comprehensive error handling and monitoring"""
    try:
        logger.info("Starting production training...")
        logger.info(f"Configuration: {asdict(config)}")
        
        # Initialize model and tokenizers
        model, tokenizer_en_to_ps, tokenizer_ps_to_en = create_translation_model(config)
        
        # Prepare dataset
        logger.info("Loading and preparing dataset...")
        full_dataset = ProductionTranslationDataset(
            config.csv_file,
            tokenizer_en_to_ps,
            tokenizer_ps_to_en,
            config
        )
        
        # Split dataset
        if config.use_all_data:
            # Use all data for training (no validation split)
            train_dataset = full_dataset
            val_dataset = full_dataset
            logger.info(f"Using all {len(train_dataset)} samples for training")
        else:
            # Split dataset for validation
            train_size = int((1 - config.validation_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Training arguments with production settings
        training_args = Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16 and config.device == "cuda",
            dataloader_pin_memory=config.dataloader_pin_memory,
            logging_dir=f'{config.output_dir}/logs',
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy="steps",
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_num_workers=config.num_workers,
            remove_unused_columns=False,
            seed=config.seed,
            predict_with_generate=True,
            save_strategy="steps",
            logging_first_step=True,
        )
        
        # Initialize trainer with custom callback
        trainer_callback = ProductionTrainerCallback(config)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        )
        
        # Add custom callback
        trainer.add_callback(trainer_callback)
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and tokenizers with comprehensive metadata
        logger.info("Saving production model...")
        
        # Create model metadata
        metadata = {
            "training_config": asdict(config),
            "training_completed": datetime.now().isoformat(),
            "dataset_size": len(full_dataset),
            "model_architecture": "MarianMT",
            "source_languages": ["English", "Pashto"],
            "target_languages": ["Pashto", "English"],
            "best_eval_loss": trainer_callback.best_eval_loss
        }
        
        # Save model components
        model.save_pretrained(config.output_dir)
        tokenizer_en_to_ps.save_pretrained(f"{config.output_dir}/tokenizer_en_to_ps")
        tokenizer_ps_to_en.save_pretrained(f"{config.output_dir}/tokenizer_ps_to_en")
        
        # Save with joblib for production use
        joblib.dump(model, f"{config.output_dir}/translation_model.joblib")
        joblib.dump(tokenizer_en_to_ps, f"{config.output_dir}/tokenizer_en_to_ps.joblib")
        joblib.dump(tokenizer_ps_to_en, f"{config.output_dir}/tokenizer_ps_to_en.joblib")
        
        # Save metadata
        with open(f"{config.output_dir}/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to {config.output_dir}")
        logger.info(f"Best evaluation loss: {trainer_callback.best_eval_loss:.6f}")
        
        return model, tokenizer_en_to_ps, tokenizer_ps_to_en
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def translate_text(text: str, model: MarianMTModel, tokenizer: MarianTokenizer, 
                  target_lang: str = 'en', max_length: int = 512) -> str:
    """Production-ready translation with error handling"""
    try:
        # Validate input
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
            
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, 
                                   num_beams=4, early_stopping=True)
            
        # Decode output
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        logger.error(traceback.format_exc())
        return "Translation failed"

def load_model(model_path: str) -> Tuple[DualTranslationModel, MarianTokenizer, MarianTokenizer]:
    """Load trained model for inference"""
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load with joblib
        model = joblib.load(f"{model_path}/translation_model.joblib")