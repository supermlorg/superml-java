package org.superml.examples.transformers;

import org.superml.transformers.models.TransformerModel;
import java.util.*;

/**
 * Text Prediction Example: "What is my" → next word prediction
 * 
 * This example demonstrates:
 * 1. Text tokenization and vocabulary creation
 * 2. Training a transformer for next-word prediction
 * 3. Generating predictions for "What is my ?"
 * 4. Understanding how transformers process sequential text data
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TextPredictionExample {
    
    // Simple vocabulary for demonstration
    private static final Map<String, Integer> WORD_TO_ID = new HashMap<>();
    private static final Map<Integer, String> ID_TO_WORD = new HashMap<>();
    private static final int VOCAB_SIZE = 50; // Smaller vocab for demo
    private static final int MODEL_DIM = 64;  // Smaller model for demo
    
    static {
        // Build vocabulary with common words and responses to "What is my..."
        String[] vocabulary = {
            "<PAD>", "<UNK>", "<START>", "<END>",
            "what", "is", "my", "the", "a", "an",
            // Common responses to "What is my..."
            "name", "car", "dog", "cat", "house", "book", "phone",
            "favorite", "best", "new", "old", "big", "small",
            "color", "food", "movie", "song", "friend", "job",
            "address", "number", "age", "birthday", "email",
            "hobby", "sport", "game", "music", "style", "dream",
            // Additional words
            "good", "bad", "red", "blue", "green", "white", "black",
            "happy", "sad", "fast", "slow", "hot", "cold"
        };
        
        for (int i = 0; i < vocabulary.length && i < VOCAB_SIZE; i++) {
            WORD_TO_ID.put(vocabulary[i], i);
            ID_TO_WORD.put(i, vocabulary[i]);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("🤖 Text Prediction with Transformers: 'What is my' → ?");
        System.out.println("=====================================================");
        
        try {
            // Step 1: Show vocabulary
            System.out.println("📖 Vocabulary Overview:");
            System.out.println("   Size: " + VOCAB_SIZE + " words");
            System.out.println("   Sample tokens: " + 
                Arrays.asList("what", "is", "my", "name", "car", "phone", "favorite"));
            
            // Step 2: Prepare training data
            List<String> trainingTexts = createTrainingData();
            
            System.out.println("\n📚 Training Data (" + trainingTexts.size() + " examples):");
            for (int i = 0; i < Math.min(10, trainingTexts.size()); i++) {
                System.out.println("   \"" + trainingTexts.get(i) + "\"");
            }
            if (trainingTexts.size() > 10) {
                System.out.println("   ... and " + (trainingTexts.size() - 10) + " more examples");
            }
            
            // Step 3: Convert text to numerical format
            System.out.println("\n🔢 Text to Numbers Conversion:");
            DataSet dataset = prepareDataset(trainingTexts);
            
            System.out.println("   Example: \"what is my name\" →");
            System.out.println("     Input:  [what=" + WORD_TO_ID.get("what") + 
                             ", is=" + WORD_TO_ID.get("is") + 
                             ", my=" + WORD_TO_ID.get("my") + ", 0, 0, ...]");
            System.out.println("     Target: name=" + WORD_TO_ID.get("name"));
            
            // Step 4: Create and configure transformer
            System.out.println("\n🏗️ Creating Transformer Model:");
            TransformerModel transformer = createTextPredictionModel();
            
            System.out.println("   ✅ Model created successfully");
            System.out.println("   - Architecture: Encoder-Only (BERT-style for classification)");
            System.out.println("   - Layers: 3 (lightweight for demo)");
            System.out.println("   - Model Dimension: " + MODEL_DIM);
            System.out.println("   - Attention Heads: 4");
            System.out.println("   - Output Classes: " + VOCAB_SIZE + " (one per word)");
            
            // Step 5: Train the model
            System.out.println("\n🎯 Training Phase:");
            trainModel(transformer, dataset);
            
            // Step 6: Test predictions
            System.out.println("\n🔮 Prediction Phase:");
            testPredictions(transformer);
            
            // Step 7: Explain the process
            System.out.println("\n🧠 How the Transformer Works:");
            explainTransformerProcess();
            
            System.out.println("\n✅ Text prediction example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static List<String> createTrainingData() {
        // Create diverse training examples for "what is my X" pattern
        return Arrays.asList(
            // Basic possessions
            "what is my name", "what is my car", "what is my dog", "what is my cat",
            "what is my house", "what is my book", "what is my phone", "what is my job",
            
            // Favorites and preferences  
            "what is my favorite color", "what is my favorite food", "what is my favorite movie",
            "what is my favorite song", "what is my favorite game", "what is my best friend",
            
            // Personal details
            "what is my age", "what is my birthday", "what is my email", "what is my address",
            "what is my number", "what is my hobby", "what is my dream",
            
            // Descriptive attributes
            "what is my new car", "what is my old book", "what is my big house",
            "what is my small phone", "what is my good friend", "what is my red car",
            
            // More complex patterns
            "what is my favorite sport", "what is my music style", "what is my happy song",
            "what is my fast car", "what is my hot food", "what is my cold drink"
        );
    }
    
    private static DataSet prepareDataset(List<String> texts) {
        List<double[]> inputs = new ArrayList<>();
        List<Double> targets = new ArrayList<>();
        
        System.out.println("   Processing " + texts.size() + " training examples...");
        
        int validExamples = 0;
        for (String text : texts) {
            String[] words = text.toLowerCase().split("\\s+");
            
            if (words.length >= 4) { // Need at least "what is my [target]"
                // Create input: "what is my" padded to MODEL_DIM
                double[] input = new double[MODEL_DIM];
                
                // First 3 positions: "what is my"
                for (int i = 0; i < 3; i++) {
                    Integer wordId = WORD_TO_ID.get(words[i]);
                    if (wordId != null) {
                        input[i] = wordId;
                    } else {
                        input[i] = WORD_TO_ID.get("<UNK>"); // Unknown word
                    }
                }
                
                // Pad remaining positions with <PAD> token
                for (int i = 3; i < MODEL_DIM; i++) {
                    input[i] = WORD_TO_ID.get("<PAD>");
                }
                
                // Target: the word after "what is my"
                Integer targetId = WORD_TO_ID.get(words[3]);
                if (targetId != null) {
                    inputs.add(input);
                    targets.add((double) targetId);
                    validExamples++;
                }
            }
        }
        
        System.out.println("   ✅ Processed " + validExamples + " valid examples");
        
        // Convert to arrays
        double[][] X = inputs.toArray(new double[0][]);
        double[] y = targets.stream().mapToDouble(Double::doubleValue).toArray();
        
        return new DataSet(X, y);
    }
    
    private static TransformerModel createTextPredictionModel() {
        // Create encoder-only model for classification
        // Each word in vocabulary is a "class"
        return TransformerModel.createEncoderOnly(
            3,          // 3 layers (lightweight)
            MODEL_DIM,  // 64 dimensions
            4,          // 4 attention heads  
            VOCAB_SIZE  // number of classes = vocabulary size
        );
    }
    
    private static void trainModel(TransformerModel model, DataSet dataset) {
        long startTime = System.currentTimeMillis();
        
        System.out.println("   📊 Dataset: " + dataset.X.length + " examples, " + 
                          dataset.X[0].length + " features per example");
        
        // Train the model
        model.fit(dataset.X, dataset.y);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.println("   ⏱️ Training completed in " + trainingTime + " ms");
        
        // Calculate training accuracy
        double accuracy = model.score(dataset.X, dataset.y);
        System.out.println("   📈 Training accuracy: " + String.format("%.1f%%", accuracy * 100));
    }
    
    private static void testPredictions(TransformerModel model) {
        // Test the classic "What is my" query
        System.out.println("   🎯 Testing: \"What is my\" → ?");
        
        // Prepare input
        double[] queryInput = new double[MODEL_DIM];
        queryInput[0] = WORD_TO_ID.get("what");
        queryInput[1] = WORD_TO_ID.get("is");
        queryInput[2] = WORD_TO_ID.get("my");
        // Rest filled with <PAD> tokens
        for (int i = 3; i < MODEL_DIM; i++) {
            queryInput[i] = WORD_TO_ID.get("<PAD>");
        }
        
        // Get predictions
        double[] prediction = model.predict(new double[][]{queryInput});
        double[][] probabilities = model.predictProba(new double[][]{queryInput});
        
        // Show most likely prediction
        int predictedWordId = (int) prediction[0];
        String predictedWord = ID_TO_WORD.getOrDefault(predictedWordId, "<UNKNOWN>");
        double confidence = probabilities[0][predictedWordId];
        
        System.out.println("\n   🎯 Most likely completion: \"What is my " + predictedWord + "\"");
        System.out.println("   📊 Confidence: " + String.format("%.1f%%", confidence * 100));
        
        // Show top 5 predictions
        System.out.println("\n   📈 Top 5 Predictions:");
        List<WordProbability> topPredictions = getTopPredictions(probabilities[0], 5);
        
        for (int i = 0; i < topPredictions.size(); i++) {
            WordProbability wp = topPredictions.get(i);
            System.out.printf("      %d. \"What is my %s\" (%.1f%%)\n", 
                i + 1, wp.word, wp.probability * 100);
        }
        
        // Test with different context
        System.out.println("\n   🎪 Testing variations:");
        testVariation(model, "Testing other patterns (conceptual)");
    }
    
    private static void testVariation(TransformerModel model, String note) {
        System.out.println("      " + note);
        System.out.println("      - \"What is my favorite\" → (would predict: color, food, movie...)");
        System.out.println("      - \"What is my new\" → (would predict: car, book, phone...)");
        System.out.println("      - \"What is my best\" → (would predict: friend, song, game...)");
    }
    
    private static List<WordProbability> getTopPredictions(double[] probs, int topK) {
        List<WordProbability> predictions = new ArrayList<>();
        
        // Collect all valid predictions
        for (int i = 0; i < probs.length; i++) {
            if (ID_TO_WORD.containsKey(i) && probs[i] > 0.001) { // Filter very low probabilities
                String word = ID_TO_WORD.get(i);
                // Skip special tokens in top predictions
                if (!word.startsWith("<")) {
                    predictions.add(new WordProbability(word, probs[i]));
                }
            }
        }
        
        // Sort by probability (highest first)
        predictions.sort((a, b) -> Double.compare(b.probability, a.probability));
        
        // Return top K
        return predictions.subList(0, Math.min(topK, predictions.size()));
    }
    
    private static void explainTransformerProcess() {
        System.out.println("   🔍 Step-by-step Process:");
        System.out.println("   1. 📝 Tokenization:");
        System.out.println("      \"what is my\" → [" + WORD_TO_ID.get("what") + 
                          ", " + WORD_TO_ID.get("is") + ", " + WORD_TO_ID.get("my") + ", 0, 0, ...]");
        
        System.out.println("   2. 🔢 Embedding:");
        System.out.println("      Each token ID becomes a " + MODEL_DIM + "-dimensional vector");
        System.out.println("      [4, 5, 6] → [[0.1, 0.4, ...], [0.3, 0.8, ...], [0.2, 0.1, ...]]");
        
        System.out.println("   3. 📍 Positional Encoding:");
        System.out.println("      Add position information so model knows word order");
        System.out.println("      Position 0: \"what\", Position 1: \"is\", Position 2: \"my\"");
        
        System.out.println("   4. 🧠 Multi-Head Attention:");
        System.out.println("      - Head 1: \"my\" attends to \"what\" (question context)");
        System.out.println("      - Head 2: \"my\" attends to \"is\" (grammatical structure)");
        System.out.println("      - Head 3: All words attend to each other (full context)");
        System.out.println("      - Head 4: Focus on \"my\" (possessive → noun likely follows)");
        
        System.out.println("   5. 🔄 Feed Forward:");
        System.out.println("      Process attended representations through neural network");
        
        System.out.println("   6. 📊 Classification:");
        System.out.println("      Output probability distribution over " + VOCAB_SIZE + " vocabulary words");
        System.out.println("      \"name\": 25%, \"car\": 18%, \"phone\": 15%, \"dog\": 12%...");
        
        System.out.println("   7. 🎯 Prediction:");
        System.out.println("      Select word with highest probability as next token");
    }
    
    // Helper classes
    static class DataSet {
        final double[][] X;
        final double[] y;
        
        DataSet(double[][] X, double[] y) {
            this.X = X;
            this.y = y;
        }
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
