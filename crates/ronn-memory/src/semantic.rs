//! Semantic Memory - Long-term knowledge graph

use crate::{MemoryId, Result};
use std::collections::{HashMap, HashSet};

/// A concept in semantic memory
#[derive(Clone, Debug)]
pub struct Concept {
    pub id: MemoryId,
    pub name: String,
    pub activation: f64,
    pub related_concepts: HashSet<MemoryId>,
}

/// Knowledge graph of concepts and relationships
pub struct ConceptGraph {
    concepts: HashMap<MemoryId, Concept>,
    relationships: HashMap<(MemoryId, MemoryId), f64>, // (from, to) -> strength
}

impl ConceptGraph {
    /// Create a new concept graph
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            relationships: HashMap::new(),
        }
    }

    /// Add a concept
    pub fn add_concept(&mut self, concept: Concept) {
        self.concepts.insert(concept.id, concept);
    }

    /// Add a relationship between concepts
    pub fn add_relationship(&mut self, from_id: MemoryId, to_id: MemoryId, strength: f64) {
        self.relationships.insert((from_id, to_id), strength);

        // Update related concepts
        if let Some(concept) = self.concepts.get_mut(&from_id) {
            concept.related_concepts.insert(to_id);
        }
        if let Some(concept) = self.concepts.get_mut(&to_id) {
            concept.related_concepts.insert(from_id);
        }
    }

    /// Get related concepts
    pub fn get_related(&self, id: MemoryId) -> Vec<&Concept> {
        self.concepts
            .get(&id)
            .map(|concept| {
                concept
                    .related_concepts
                    .iter()
                    .filter_map(|rel_id| self.concepts.get(rel_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Activate a concept (spreads activation to related concepts)
    pub fn activate(&mut self, id: MemoryId, amount: f64) {
        if let Some(concept) = self.concepts.get_mut(&id) {
            concept.activation += amount;

            // Spread activation to related concepts
            let related: Vec<MemoryId> = concept.related_concepts.iter().copied().collect();
            for rel_id in related {
                if let Some(rel_concept) = self.concepts.get_mut(&rel_id) {
                    // Spread with decay
                    rel_concept.activation += amount * 0.5;
                }
            }
        }
    }

    /// Get number of concepts
    pub fn len(&self) -> usize {
        self.concepts.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.concepts.is_empty()
    }
}

impl Default for ConceptGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Semantic memory with concept storage and activation spreading
pub struct SemanticMemory {
    graph: ConceptGraph,
}

impl SemanticMemory {
    /// Create new semantic memory
    pub fn new() -> Self {
        Self {
            graph: ConceptGraph::new(),
        }
    }

    /// Store a concept
    pub fn store_concept(&mut self, concept: Concept) -> Result<()> {
        self.graph.add_concept(concept);
        Ok(())
    }

    /// Get number of concepts
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Get the knowledge graph
    pub fn graph(&self) -> &ConceptGraph {
        &self.graph
    }

    /// Get mutable knowledge graph
    pub fn graph_mut(&mut self) -> &mut ConceptGraph {
        &mut self.graph
    }
}

impl Default for SemanticMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use super::*;

    #[test]
    fn test_concept_storage() {
        let mut sm = SemanticMemory::new();

        let concept = Concept {
            id: 1,
            name: "test".to_string(),
            activation: 0.5,
            related_concepts: HashSet::new(),
        };

        sm.store_concept(concept).unwrap();
        assert_eq!(sm.len(), 1);
    }

    #[test]
    fn test_activation_spreading() {
        let mut graph = ConceptGraph::new();

        // Create connected concepts
        let c1 = Concept {
            id: 1,
            name: "A".to_string(),
            activation: 0.0,
            related_concepts: HashSet::new(),
        };
        let c2 = Concept {
            id: 2,
            name: "B".to_string(),
            activation: 0.0,
            related_concepts: HashSet::new(),
        };

        graph.add_concept(c1);
        graph.add_concept(c2);
        graph.add_relationship(1, 2, 0.8);

        // Activate first concept
        graph.activate(1, 1.0);

        // Check activation spread
        assert!(graph.concepts.get(&1).unwrap().activation >= 1.0);
        assert!(graph.concepts.get(&2).unwrap().activation > 0.0);
    }
}
