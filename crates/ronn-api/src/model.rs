use crate::error::{Error, Result};
use crate::session::{SessionBuilder, SessionOptions};
use ronn_onnx::{LoadedModel, ModelLoader};
use std::path::Path;
use std::sync::Arc;
use tracing::info;

/// Represents a loaded ML model
pub struct Model {
    inner: Arc<LoadedModel>,
}

impl Model {
    /// Load a model from an ONNX file
    ///
    /// # Example
    /// ```no_run
    /// use ronn_api::Model;
    ///
    /// let model = Model::load("model.onnx")?;
    /// # Ok::<(), ronn_api::Error>(())
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Loading model from: {:?}", path.as_ref());
        let loaded = ModelLoader::load_from_file(path)?;
        Ok(Self {
            inner: Arc::new(loaded),
        })
    }

    /// Load a model from bytes
    ///
    /// # Example
    /// ```no_run
    /// use ronn_api::Model;
    ///
    /// let bytes = std::fs::read("model.onnx")?;
    /// let model = Model::from_bytes(&bytes)?;
    /// # Ok::<(), ronn_api::Error>(())
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        info!("Loading model from bytes ({} bytes)", bytes.len());
        let loaded = ModelLoader::load_from_bytes(bytes)?;
        Ok(Self {
            inner: Arc::new(loaded),
        })
    }

    /// Create an inference session with default options
    ///
    /// # Example
    /// ```no_run
    /// use ronn_api::Model;
    ///
    /// let model = Model::load("model.onnx")?;
    /// let session = model.create_session_default()?;
    /// # Ok::<(), ronn_api::Error>(())
    /// ```
    pub fn create_session_default(&self) -> Result<crate::session::InferenceSession> {
        self.create_session(SessionOptions::default())
    }

    /// Create an inference session with custom options
    ///
    /// # Example
    /// ```no_run
    /// use ronn_api::{Model, SessionOptions, OptimizationLevel};
    /// use ronn_providers::ProviderType;
    ///
    /// let model = Model::load("model.onnx")?;
    /// let options = SessionOptions::new()
    ///     .with_optimization_level(OptimizationLevel::O3)
    ///     .with_provider(ProviderType::GPU);
    /// let session = model.create_session(options)?;
    /// # Ok::<(), ronn_api::Error>(())
    /// ```
    pub fn create_session(&self, options: SessionOptions) -> Result<crate::session::InferenceSession> {
        SessionBuilder::new(self.inner.clone(), options).build()
    }

    /// Get model metadata
    pub fn producer_name(&self) -> Option<&str> {
        self.inner.producer_name.as_deref()
    }

    /// Get IR version
    pub fn ir_version(&self) -> i64 {
        self.inner.ir_version
    }

    /// Get input names
    pub fn input_names(&self) -> Vec<&str> {
        self.inner.inputs().iter().map(|i| i.name.as_str()).collect()
    }

    /// Get output names
    pub fn output_names(&self) -> Vec<&str> {
        self.inner
            .outputs()
            .iter()
            .map(|o| o.name.as_str())
            .collect()
    }
}
