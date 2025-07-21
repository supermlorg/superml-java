package org.superml.examples.transformers;

import org.superml.transformers.models.TransformerModel;
import java.util.*;

/**
 * Advanced Text Prediction Example with Realistic Scenarios
 * 
 * This example shows:
 * 1. More sophisticated text prediction patterns
 * 2. Multiple context variations ("What is my favorite", "What is my new", etc.)
 * 3. Probability analysis and explanation
 * 4. Comparison with different training approaches
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class AdvancedTextPredictionExample {
    
    // Enhanced vocabulary with more realistic word patterns
    private static final Map<String, Integer> WORD_TO_ID = new HashMap<>();
    private static final Map<Integer, String> ID_TO_WORD = new HashMap<>();
    private static final int VOCAB_SIZE = 80;
    private static final int MODEL_DIM = 128; // Larger model for better performance
    
    static {
        String[] vocabulary = {
            // Special tokens
            "<PAD>", "<UNK>", "<START>", "<END>",
            
            // Query words
            "what", "is", "my", "the", "a", "an", "your", "his", "her",
            
            // Common nouns (direct objects)
            "name", "car", "dog", "cat", "house", "book", "phone", "job", "age",
            "address", "number", "email", "birthday", "hobby", "dream",
            
            // Adjective + noun combinations
            "favorite", "best", "worst", "new", "old", "big", "small", "good", "bad",
            "color", "food", "movie", "song", "friend", "sport", "game", "music",
            "restaurant", "place", "memory", "achievement", "goal",
            
            // Colors and descriptors
            "red", "blue", "green", "white", "black", "yellow", "purple", "orange",
            "fast", "slow", "hot", "cold", "happy", "sad", "funny", "scary",
            
            // Activities and interests  
            "reading", "cooking", "swimming", "running", "dancing", "singing",
            "traveling", "writing", "painting", "programming", "gaming",
            
            // Personal items
            "laptop", "watch", "shoes", "shirt", "glasses", "bag", "key", "wallet"
        };
        
        for (int i = 0; i < Math.min(vocabulary.length, VOCAB_SIZE); i++) {
            WORD_TO_ID.put(vocabulary[i], i);
            ID_TO_WORD.put(i, vocabulary[i]);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("🚀 Advanced Text Prediction with Transformers");
        System.out.println("============================================");
        
        try {
            // Demo 1: Basic "What is my" prediction
            demonstrateBasicPrediction();
            
            // Demo 2: Context-aware predictions
            demonstrateContextualPredictions();
            
            // Demo 3: Pattern analysis
            analyzePatterns();
            
            System.out.println("\n🎯 Advanced text prediction completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void demonstrateBasicPrediction() {
        System.out.println("\n1. 📝 Basic Text Prediction: 'What is my' → ?");
        System.out.println("===============================================");
        
        // Create comprehensive training data
        List<String> basicTraining = createBasicTrainingData();
        
        System.out.println("📚 Training data overview:");
        System.out.println("   Total examples: " + basicTraining.size());
        System.out.println("   Sample patterns:");
        for (int i = 0; i < Math.min(8, basicTraining.size()); i++) {
            System.out.println("     \"" + basicTraining.get(i) + "\"");
        }
        
        // Train and test basic model
        TransformerModel basicModel = trainModel(basicTraining, "basic");
        testQuery(basicModel, new String[]{"what", "is", "my"}, "Basic Model");
    }
    
    private static void demonstrateContextualPredictions() {
        System.out.println("\n2. 🧠 Context-Aware Predictions");
        System.out.println("==============================");
        
        // Create models for different contexts
        Map<String, List<String>> contextualData = createContextualTrainingData();
        
        for (Map.Entry<String, List<String>> entry : contextualData.entrySet()) {
            String context = entry.getKey();
            List<String> trainingData = entry.getValue();
            
            System.out.println("\n   🎯 Context: " + context.toUpperCase());
            System.out.println("   Training examples: " + trainingData.size());
            
            // Show sample training data
            for (int i = 0; i < Math.min(3, trainingData.size()); i++) {
                System.out.println("     \"" + trainingData.get(i) + "\"");
            }
            
            // Train specialized model
            TransformerModel contextModel = trainModel(trainingData, context);
            
            // Test with appropriate query
            String[] query = getQueryForContext(context);
            testQuery(contextModel, query, context + " Model");
        }
    }
    
    private static void analyzePatterns() {
        System.out.println("\n3. 📊 Pattern Analysis");
        System.out.println("=====================");
        
        // Create comprehensive dataset
        List<String> allTrainingData = createComprehensiveTrainingData();
        TransformerModel comprehensiveModel = trainModel(allTrainingData, "comprehensive");
        
        System.out.println("\n   🔍 Testing various query patterns:");
        
        // Test different patterns
        String[][] testPatterns = {
            {"what", "is", "my"},
            {"what", "is", "my", "favorite"},
            {"what", "is", "my", "new"},
            {"what", "is", "my", "best"}
        };
        
        for (String[] pattern : testPatterns) {
            System.out.println("\n   📝 Query: \"" + String.join(" ", pattern) + " ?\"");
            testQueryWithAnalysis(comprehensiveModel, pattern);
        }
        
        // Explain the transformer's behavior
        explainTransformerBehavior();
    }
    
    private static List<String> createBasicTrainingData() {
        return Arrays.asList(
            "what is my name", "what is my car", "what is my dog", "what is my cat",
            "what is my house", "what is my phone", "what is my job", "what is my age",
            "what is my book", "what is my laptop", "what is my watch", "what is my key"
        );
    }
    
    private static Map<String, List<String>> createContextualTrainingData() {
        Map<String, List<String>> contextData = new HashMap<>();
        
        contextData.put("favorites", Arrays.asList(
            "what is my favorite color", "what is my favorite food", "what is my favorite movie",
            "what is my favorite song", "what is my favorite sport", "what is my favorite game",
            "what is my favorite restaurant", "what is my favorite place", "what is my favorite book"
        ));
        
        contextData.put("possessions", Arrays.asList(
            "what is my new car", "what is my new phone", "what is my new laptop",
            "what is my old house", "what is my big dog", "what is my small cat",
            "what is my red car", "what is my blue shirt", "what is my black shoes"
        ));
        
        contextData.put("relationships", Arrays.asList(
            "what is my best friend", "what is my good friend", "what is my old friend",
            "what is my happy memory", "what is my sad memory", "what is my funny memory"
        ));
        
        return contextData;
    }
    
    private static List<String> createComprehensiveTrainingData() {
        List<String> comprehensive = new ArrayList<>();
        
        // Combine all patterns
        comprehensive.addAll(createBasicTrainingData());
        
        Map<String, List<String>> contextual = createContextualTrainingData();
        for (List<String> data : contextual.values()) {
            comprehensive.addAll(data);
        }
        
        // Add more complex patterns
        comprehensive.addAll(Arrays.asList(
            "what is my programming language", "what is my cooking style",
            "what is my reading habit", "what is my travel destination",
            "what is my workout routine", "what is my study method"
        ));
        
        return comprehensive;
    }
    
    private static String[] getQueryForContext(String context) {
        switch (context) {
            case "favorites":
                return new String[]{"what", "is", "my", "favorite"};
            case "possessions":
                return new String[]{"what", "is", "my", "new"};
            case "relationships":
                return new String[]{"what", "is", "my", "best"};
            default:
                return new String[]{"what", "is", "my"};
        }
    }
    
    private static TransformerModel trainModel(List<String> trainingData, String modelName) {
        // Convert text to numerical format
        List<double[]> inputs = new ArrayList<>();
        List<Double> targets = new ArrayList<>();
        
        for (String text : trainingData) {
            String[] words = text.toLowerCase().split("\\s+");
            
            if (words.length >= 4) {
                // Create input sequence
                double[] input = new double[MODEL_DIM];
                
                // Fill in the context words (up to MODEL_DIM)
                int numContext = Math.min(words.length - 1, MODEL_DIM);
                for (int i = 0; i < numContext; i++) {
                    Integer wordId = WORD_TO_ID.get(words[i]);
                    input[i] = wordId != null ? wordId : WORD_TO_ID.get("<UNK>");
                }
                
                // Pad remaining
                for (int i = numContext; i < MODEL_DIM; i++) {
                    input[i] = WORD_TO_ID.get("<PAD>");
                }
                
                // Target: last word
                Integer targetId = WORD_TO_ID.get(words[words.length - 1]);
                if (targetId != null) {
                    inputs.add(input);
                    targets.add((double) targetId);
                }
            }
        }
        
        // Convert to arrays
        double[][] X = inputs.toArray(new double[0][]);
        double[] y = targets.stream().mapToDouble(Double::doubleValue).toArray();
        
        // Create and train model
        TransformerModel model = TransformerModel.createEncoderOnly(
            4,          // 4 layers
            MODEL_DIM,  // 128 dimensions
            8,          // 8 attention heads
            VOCAB_SIZE  // vocabulary size
        );
        
        System.out.println("   🔧 Training " + modelName + " model (" + X.length + " examples)...");
        model.fit(X, y);
        
        return model;
    }
    
    private static void testQuery(TransformerModel model, String[] queryWords, String modelName) {
        // Prepare input
        double[] input = new double[MODEL_DIM];
        
        for (int i = 0; i < queryWords.length && i < MODEL_DIM; i++) {
            Integer wordId = WORD_TO_ID.get(queryWords[i]);
            input[i] = wordId != null ? wordId : WORD_TO_ID.get("<UNK>");
        }
        
        // Pad remaining
        for (int i = queryWords.length; i < MODEL_DIM; i++) {
            input[i] = WORD_TO_ID.get("<PAD>");
        }
        
        // Get predictions
        double[] prediction = model.predict(new double[][]{input});
        double[][] probabilities = model.predictProba(new double[][]{input});
        
        // Show results
        String query = String.join(" ", queryWords);
        int predictedId = (int) prediction[0];
        String predictedWord = ID_TO_WORD.getOrDefault(predictedId, "<UNKNOWN>");
        double confidence = probabilities[0][predictedId];
        
        System.out.println("   🎯 " + modelName + ": \"" + query + " " + predictedWord + "\" (" + 
                          String.format("%.1f%%", confidence * 100) + " confidence)");
    }
    
    private static void testQueryWithAnalysis(TransformerModel model, String[] queryWords) {
        double[] input = new double[MODEL_DIM];
        
        for (int i = 0; i < queryWords.length && i < MODEL_DIM; i++) {
            Integer wordId = WORD_TO_ID.get(queryWords[i]);
            input[i] = wordId != null ? wordId : WORD_TO_ID.get("<UNK>");
        }
        
        for (int i = queryWords.length; i < MODEL_DIM; i++) {
            input[i] = WORD_TO_ID.get("<PAD>");
        }
        
        double[][] probabilities = model.predictProba(new double[][]{input});
        
        // Get top 3 predictions
        List<WordProbability> topPredictions = getTopPredictions(probabilities[0], 3);
        
        System.out.println("     Top predictions:");
        for (int i = 0; i < topPredictions.size(); i++) {
            WordProbability wp = topPredictions.get(i);
            System.out.printf("       %d. %s (%.1f%%)\n", i + 1, wp.word, wp.probability * 100);
        }
    }
    
    private static void explainTransformerBehavior() {
        System.out.println("\n   🧠 How the Transformer Learns Patterns:");
        System.out.println("   ========================================");
        
        System.out.println("   📚 Training Process:");
        System.out.println("     1. Sees many examples: 'what is my X'");
        System.out.println("     2. Learns that after 'what is my', certain words are common");
        System.out.println("     3. Attention heads specialize in different patterns:");
        System.out.println("        - Head 1: Grammar (subject-verb agreement)");
        System.out.println("        - Head 2: Semantics (meaningful word combinations)");
        System.out.println("        - Head 3: Context (question → expected answer type)");
        System.out.println("        - Head 4: Position (word order importance)");
        
        System.out.println("\n   🎯 Prediction Process:");
        System.out.println("     1. Input: 'what is my favorite'");
        System.out.println("     2. Attention: 'favorite' strongly attends to 'what'");
        System.out.println("     3. Context understanding: This is asking for a preference");
        System.out.println("     4. Probability distribution: favors words like 'color', 'food', 'movie'");
        System.out.println("     5. Output: Most likely completion based on training patterns");
        
        System.out.println("\n   💡 Key Insights:");
        System.out.println("     • More training data = better predictions");
        System.out.println("     • Context words ('favorite', 'new', 'best') change predictions");
        System.out.println("     • Model learns statistical patterns from training examples");
        System.out.println("     • Attention mechanism allows focus on relevant parts");
    }
    
    private static List<WordProbability> getTopPredictions(double[] probs, int topK) {
        List<WordProbability> predictions = new ArrayList<>();
        
        for (int i = 0; i < probs.length; i++) {
            if (ID_TO_WORD.containsKey(i) && probs[i] > 0.001) {
                String word = ID_TO_WORD.get(i);
                if (!word.startsWith("<")) {
                    predictions.add(new WordProbability(word, probs[i]));
                }
            }
        }
        
        predictions.sort((a, b) -> Double.compare(b.probability, a.probability));
        return predictions.subList(0, Math.min(topK, predictions.size()));
    }
    
    static class WordProbability {
        final String word;
        final double probability;
        
        WordProbability(String word, double probability) {
            this.word = word;
            this.probability = probability;
        }
    }
}
