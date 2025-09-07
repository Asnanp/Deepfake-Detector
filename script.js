// Professional DeepFake Detector
class DeepFakeDetector {
  constructor() {
    this.currentFile = null;
    this.init();
  }

  init() {
    console.log('DeepFake Detector initializing...');
    this.setupEventListeners();
    this.updateNavStatus('Ready');
    console.log('DeepFake Detector initialized successfully');
  }

  setupEventListeners() {
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea && fileInput) {
      uploadArea.addEventListener('click', () => fileInput.click());
      uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
      uploadArea.addEventListener('drop', this.handleDrop.bind(this));
      fileInput.addEventListener('change', this.handleFileSelect.bind(this));
    }
  }

  // File Handling
  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.processFile(file);
    }
  }

  handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
  }

  handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      this.processFile(files[0]);
    }
  }

  async processFile(file) {
    try {
      this.currentFile = file;
      this.updateNavStatus('Processing...');
      
      // Validate file type (image only)
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        this.showError('Please upload an image file (JPG, PNG, WebP)');
        return;
      }

      // Validate file size (50MB max)
      if (file.size > 50 * 1024 * 1024) {
        this.showError('File size must be less than 50MB');
        return;
      }

      // Show preview
      await this.displayPreview(file);
      this.showSection('preview');
      
    } catch (error) {
      console.error('File processing error:', error);
      this.showError('Error processing file. Please try again.');
    }
  }

  async displayPreview(file) {
    const previewContainer = document.getElementById('previewContainer');
    const fileInfo = document.getElementById('fileInfo');
    
    // Create image preview
    const url = URL.createObjectURL(file);
    previewContainer.innerHTML = '';
    const img = document.createElement('img');
    img.src = url;
    img.style.maxWidth = '100%';
    img.style.maxHeight = '300px';
    img.style.borderRadius = '12px';
    previewContainer.appendChild(img);

    // Update file info
    fileInfo.innerHTML = `
      <div class="info-item">
        <span class="info-label">Name:</span>
        <span class="info-value">${file.name}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Size:</span>
        <span class="info-value">${this.formatFileSize(file.size)}</span>
      </div>
      <div class="info-item">
        <span class="info-label">Type:</span>
        <span class="info-value">${file.type}</span>
      </div>
    `;
  }

  async analyzeImage() {
    if (!this.currentFile) {
      this.showError('No file selected');
      return;
    }

    try {
      this.showSection('loading');
      this.updateNavStatus('Analyzing...');
      
      const result = await this.callAPI(this.currentFile);
      this.displayResults(result);
      this.showSection('results');
      this.updateNavStatus('Analysis Complete');
      
    } catch (error) {
      console.error('Analysis error:', error);
      this.showError('Analysis failed. Please try again.');
      this.updateNavStatus('Error');
    }
  }

  async callAPI(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/detect', {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(`API Error: ${errorData}`);
    }
    
    return await response.json();
  }

  displayResults(result) {
    console.log('Analysis result:', result);
    
    // Update verdict badge
    const verdictBadge = document.getElementById('verdictBadge');
    if (verdictBadge) {
      verdictBadge.textContent = result.prediction || 'Unknown';
      verdictBadge.className = `verdict-badge ${result.is_real ? 'authentic' : 'fake'}`;
    }
    
    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceValue) {
      confidenceValue.textContent = `${result.confidence || 0}%`;
    }
    
    // Update prediction
    const predictionValue = document.getElementById('predictionValue');
    if (predictionValue) {
      predictionValue.textContent = result.prediction || 'Unknown';
    }
    
    // Update processing time
    const processingTime = document.getElementById('processingTime');
    if (processingTime) {
      const ms = typeof result.processing_time === 'number' ? Math.round(result.processing_time * 1000) : 0;
      processingTime.textContent = `${ms}ms`;
    }
    
    // Keep results minimal: do not render advanced analysis sections
  }

  displayELAAnalysis(elaData) {
    // Create or update ELA analysis section
    let elaSection = document.getElementById('elaAnalysis');
    if (!elaSection) {
      elaSection = document.createElement('div');
      elaSection.id = 'elaAnalysis';
      elaSection.className = 'analysis-section';
      elaSection.innerHTML = `
        <h4>ELA Analysis</h4>
        <div class="analysis-details">
          <div class="analysis-item">
            <span class="analysis-label">Manipulation Score:</span>
            <span class="analysis-value" id="elaManipulationScore">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">ELA Mean:</span>
            <span class="analysis-value" id="elaMean">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">ELA Std:</span>
            <span class="analysis-value" id="elaStd">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Manipulation Detected:</span>
            <span class="analysis-value" id="elaManipulationDetected">--</span>
          </div>
        </div>
      `;
      
      // Insert after results details
      const resultsDetails = document.querySelector('.results-details');
      if (resultsDetails) {
        resultsDetails.parentNode.insertBefore(elaSection, resultsDetails.nextSibling);
      }
    }
    
    // Update ELA values
    const manipulationScore = document.getElementById('elaManipulationScore');
    if (manipulationScore) {
      manipulationScore.textContent = `${(elaData.manipulation_score * 100).toFixed(1)}%`;
    }
    
    const elaMean = document.getElementById('elaMean');
    if (elaMean) {
      elaMean.textContent = elaData.ela_mean.toFixed(2);
    }
    
    const elaStd = document.getElementById('elaStd');
    if (elaStd) {
      elaStd.textContent = elaData.ela_std.toFixed(2);
    }
    
    const manipulationDetected = document.getElementById('elaManipulationDetected');
    if (manipulationDetected) {
      manipulationDetected.textContent = elaData.is_manipulated ? 'Yes' : 'No';
      manipulationDetected.className = `analysis-value ${elaData.is_manipulated ? 'manipulated' : 'clean'}`;
    }
  }

  displayNoiseAnalysis(noiseData) {
    // Create or update noise analysis section
    let noiseSection = document.getElementById('noiseAnalysis');
    if (!noiseSection) {
      noiseSection = document.createElement('div');
      noiseSection.id = 'noiseAnalysis';
      noiseSection.className = 'analysis-section';
      noiseSection.innerHTML = `
        <h4>Noise Analysis</h4>
        <div class="analysis-details">
          <div class="analysis-item">
            <span class="analysis-label">Noise Std:</span>
            <span class="analysis-value" id="noiseStd">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">SNR (dB):</span>
            <span class="analysis-value" id="snrDb">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Noise Entropy:</span>
            <span class="analysis-value" id="noiseEntropy">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Artificial Noise:</span>
            <span class="analysis-value" id="artificialNoise">--</span>
          </div>
        </div>
      `;
      
      // Insert after ELA analysis
      const elaSection = document.getElementById('elaAnalysis');
      if (elaSection) {
        elaSection.parentNode.insertBefore(noiseSection, elaSection.nextSibling);
      }
    }
    
    // Update noise values
    const noiseStd = document.getElementById('noiseStd');
    if (noiseStd) {
      noiseStd.textContent = noiseData.noise_std.toFixed(2);
    }
    
    const snrDb = document.getElementById('snrDb');
    if (snrDb) {
      snrDb.textContent = noiseData.snr_db.toFixed(1);
    }
    
    const noiseEntropy = document.getElementById('noiseEntropy');
    if (noiseEntropy) {
      noiseEntropy.textContent = noiseData.noise_entropy.toFixed(2);
    }
    
    const artificialNoise = document.getElementById('artificialNoise');
    if (artificialNoise) {
      artificialNoise.textContent = noiseData.is_artificial_noise ? 'Yes' : 'No';
      artificialNoise.className = `analysis-value ${noiseData.is_artificial_noise ? 'manipulated' : 'clean'}`;
    }
  }

  displayColorAnalysis(colorData) {
    // Create or update color analysis section
    let colorSection = document.getElementById('colorAnalysis');
    if (!colorSection) {
      colorSection = document.createElement('div');
      colorSection.id = 'colorAnalysis';
      colorSection.className = 'analysis-section';
      colorSection.innerHTML = `
        <h4>Color Distribution Analysis</h4>
        <div class="analysis-details">
          <div class="analysis-item">
            <span class="analysis-label">R/G Ratio:</span>
            <span class="analysis-value" id="rgRatio">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">R/B Ratio:</span>
            <span class="analysis-value" id="rbRatio">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Color Inconsistency:</span>
            <span class="analysis-value" id="colorInconsistency">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Color Manipulated:</span>
            <span class="analysis-value" id="colorManipulated">--</span>
          </div>
        </div>
      `;
      
      // Insert after noise analysis
      const noiseSection = document.getElementById('noiseAnalysis');
      if (noiseSection) {
        noiseSection.parentNode.insertBefore(colorSection, noiseSection.nextSibling);
      }
    }
    
    // Update color values
    const rgRatio = document.getElementById('rgRatio');
    if (rgRatio) {
      rgRatio.textContent = colorData.rg_ratio.toFixed(2);
    }
    
    const rbRatio = document.getElementById('rbRatio');
    if (rbRatio) {
      rbRatio.textContent = colorData.rb_ratio.toFixed(2);
    }
    
    const colorInconsistency = document.getElementById('colorInconsistency');
    if (colorInconsistency) {
      colorInconsistency.textContent = colorData.color_inconsistency.toFixed(2);
    }
    
    const colorManipulated = document.getElementById('colorManipulated');
    if (colorManipulated) {
      colorManipulated.textContent = colorData.is_color_manipulated ? 'Yes' : 'No';
      colorManipulated.className = `analysis-value ${colorData.is_color_manipulated ? 'manipulated' : 'clean'}`;
    }
  }

  displayCompressionAnalysis(compressionData) {
    // Create or update compression analysis section
    let compressionSection = document.getElementById('compressionAnalysis');
    if (!compressionSection) {
      compressionSection = document.createElement('div');
      compressionSection.id = 'compressionAnalysis';
      compressionSection.className = 'analysis-section';
      compressionSection.innerHTML = `
        <h4>Compression Analysis</h4>
        <div class="analysis-details">
          <div class="analysis-item">
            <span class="analysis-label">File Size:</span>
            <span class="analysis-value" id="fileSize">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Compression Ratio:</span>
            <span class="analysis-value" id="compressionRatio">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Quality Estimate:</span>
            <span class="analysis-value" id="qualityEstimate">--</span>
          </div>
          <div class="analysis-item">
            <span class="analysis-label">Has Artifacts:</span>
            <span class="analysis-value" id="hasArtifacts">--</span>
          </div>
        </div>
      `;
      
      // Insert after color analysis
      const colorSection = document.getElementById('colorAnalysis');
      if (colorSection) {
        colorSection.parentNode.insertBefore(compressionSection, colorSection.nextSibling);
      }
    }
    
    // Update compression values
    const fileSize = document.getElementById('fileSize');
    if (fileSize) {
      const sizeKB = (compressionData.file_size_bytes / 1024).toFixed(1);
      fileSize.textContent = `${sizeKB} KB`;
    }
    
    const compressionRatio = document.getElementById('compressionRatio');
    if (compressionRatio) {
      compressionRatio.textContent = `${(compressionData.compression_ratio * 100).toFixed(1)}%`;
    }
    
    const qualityEstimate = document.getElementById('qualityEstimate');
    if (qualityEstimate) {
      qualityEstimate.textContent = compressionData.quality_estimate;
      qualityEstimate.className = `analysis-value ${compressionData.quality_estimate.toLowerCase()}`;
    }
    
    const hasArtifacts = document.getElementById('hasArtifacts');
    if (hasArtifacts) {
      hasArtifacts.textContent = compressionData.has_compression_artifacts ? 'Yes' : 'No';
      hasArtifacts.className = `analysis-value ${compressionData.has_compression_artifacts ? 'manipulated' : 'clean'}`;
    }
  }

  resetAnalysis() {
    this.currentFile = null;
    this.showSection('upload');
    this.updateNavStatus('Ready');
    
    // Reset file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
      fileInput.value = '';
    }
  }

  showSection(sectionId) {
    const sections = ['upload', 'preview', 'loading', 'results'];
    sections.forEach(id => {
      const section = document.getElementById(`${id}Section`);
      if (section) {
        section.style.display = id === sectionId ? 'block' : 'none';
      }
    });
  }

  updateNavStatus(status) {
    const navStatus = document.getElementById('navStatus');
    if (navStatus) {
      navStatus.textContent = status;
      
      // Update status color
      navStatus.className = 'nav-status';
      if (status === 'Ready') {
        navStatus.style.background = '#10b981';
      } else if (status === 'Processing...' || status === 'Analyzing...') {
        navStatus.style.background = '#f59e0b';
      } else if (status === 'Analysis Complete') {
        navStatus.style.background = '#10b981';
      } else if (status === 'Error') {
        navStatus.style.background = '#ef4444';
      }
    }
  }

  showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ef4444;
      color: white;
      padding: 1rem 1.5rem;
      border-radius: 12px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
      z-index: 1000;
      font-weight: 500;
      max-width: 400px;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Remove after 5 seconds
    setTimeout(() => {
      if (document.body.contains(errorDiv)) {
        errorDiv.remove();
      }
    }, 5000);
    
    // Show upload section
    this.showSection('upload');
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
  window.deepFakeDetector = new DeepFakeDetector();
});