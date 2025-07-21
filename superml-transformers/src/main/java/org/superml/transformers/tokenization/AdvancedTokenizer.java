package org.superml.transformers.tokenization;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Enhanced Text Tokenization System
 * 
 * Implements sophisticated tokenization techniques:
 * 1. Basic word-level tokenization
 * 2. Byte-Pair Encoding (BPE) simulation
 * 3. Subword tokenization
 * 4. Special token handling ([CLS], [SEP], [PAD], [UNK])
 * 5. Vocabulary management and frequency analysis
 * 
 * This dramatically improves upon our simple tokenization in text prediction examples.
 * 
 * @author SuperML Team  
 * @version 2.1.0
 */
public class AdvancedTokenizer {
    
    // Special tokens
    public static final String PAD_TOKEN = "[PAD]";
    public static final String UNK_TOKEN = "[UNK]";  
    public static final String CLS_TOKEN = "[CLS]";
    public static final String SEP_TOKEN = "[SEP]";
    public static final String MASK_TOKEN = "[MASK]";
    
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final int maxVocabSize;
    private final boolean enableSubword;
    private final Pattern tokenPattern;
    
    // BPE-like subword pairs
    private final Map<String, Integer> subwordPairs;
    
    public AdvancedTokenizer() {
        this(10000, true);
    }
    
    public AdvancedTokenizer(int maxVocabSize, boolean enableSubword) {
        this.maxVocabSize = maxVocabSize;
        this.enableSubword = enableSubword;
        this.vocab = new LinkedHashMap<>();
        this.reverseVocab = new HashMap<>();
        this.subwordPairs = new HashMap<>();
        
        // Regex pattern for tokenization (handles punctuation, contractions)
        this.tokenPattern = Pattern.compile(
            "\\b\\w+(?:'\\w+)?\\b|[.!?;,]|[\"']"
        );
        
        // Initialize special tokens
        initializeSpecialTokens();
        
        System.out.println("🔤 Advanced Tokenizer Initialized");
        System.out.printf("   Max Vocab Size: %,d\n", maxVocabSize);
        System.out.printf("   Subword Enabled: %s\n", enableSubword ? "✅" : "❌");
    }
    
    /**
     * Build vocabulary from training corpus.
     */
    public void buildVocabulary(List<String> corpus) {
        System.out.println("\n📚 Building Vocabulary from Corpus");
        System.out.printf("   Training Documents: %,d\n", corpus.size());
        
        // Count token frequencies
        Map<String, Integer> tokenCounts = new HashMap<>();
        int totalTokens = 0;
        
        for (String document : corpus) {
            List<String> tokens = basicTokenize(document);
            totalTokens += tokens.size();
            
            for (String token : tokens) {
                tokenCounts.put(token, tokenCounts.getOrDefault(token, 0) + 1);
            }
        }
        
        System.out.printf("   Total Tokens: %,d\n", totalTokens);
        System.out.printf("   Unique Tokens: %,d\n", tokenCounts.size());
        
        // Sort by frequency and add to vocabulary
        List<Map.Entry<String, Integer>> sortedTokens = new ArrayList<>(tokenCounts.entrySet());
        sortedTokens.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        // Add most frequent tokens up to max vocab size (reserve space for special tokens)
        int availableSpace = maxVocabSize - vocab.size();
        int tokensAdded = 0;
        
        for (Map.Entry<String, Integer> entry : sortedTokens) {
            if (tokensAdded >= availableSpace) break;
            
            String token = entry.getKey();
            if (!vocab.containsKey(token)) {
                addToken(token);
                tokensAdded++;
            }
        }
        
        // Build subword pairs if enabled
        if (enableSubword && tokenCounts.size() > vocab.size()) {
            buildSubwordPairs(tokenCounts);
        }
        
        System.out.printf("   Vocabulary Size: %,d\n", vocab.size());
        System.out.printf("   Subword Pairs: %,d\n", subwordPairs.size());
        
        // Show most frequent tokens
        System.out.println("   Top 10 Tokens:");
        sortedTokens.stream()
            .limit(10)
            .forEach(entry -> System.out.printf("      '%s': %,d occurrences\n", 
                entry.getKey(), entry.getValue()));
    }
    
    /**
     * Tokenize text using advanced methods.
     */
    public List<Integer> encode(String text) {
        return encode(text, false);
    }
    
    public List<Integer> encode(String text, boolean addSpecialTokens) {
        List<String> tokens = advancedTokenize(text);
        List<Integer> tokenIds = new ArrayList<>();
        
        // Add [CLS] token at beginning if requested
        if (addSpecialTokens) {
            tokenIds.add(vocab.get(CLS_TOKEN));
        }
        
        // Convert tokens to IDs
        for (String token : tokens) {
            Integer tokenId = vocab.get(token);
            if (tokenId == null) {
                // Try subword decomposition
                List<String> subtokens = decomposeSubword(token);
                for (String subtoken : subtokens) {
                    Integer subtokenId = vocab.get(subtoken);
                    tokenIds.add(subtokenId != null ? subtokenId : vocab.get(UNK_TOKEN));
                }
            } else {
                tokenIds.add(tokenId);
            }
        }
        
        // Add [SEP] token at end if requested
        if (addSpecialTokens) {
            tokenIds.add(vocab.get(SEP_TOKEN));
        }
        
        return tokenIds;
    }
    
    /**
     * Decode token IDs back to text.
     */
    public String decode(List<Integer> tokenIds) {
        StringBuilder text = new StringBuilder();
        boolean first = true;
        
        for (Integer tokenId : tokenIds) {
            String token = reverseVocab.get(tokenId);
            if (token == null) token = UNK_TOKEN;
            
            // Skip special tokens in output (except UNK)
            if (isSpecialToken(token) && !token.equals(UNK_TOKEN)) {
                continue;
            }
            
            if (!first && !isPunctuation(token)) {
                text.append(" ");
            }
            text.append(token);
            first = false;
        }
        
        return text.toString().trim();
    }
    
    /**
     * Pad or truncate sequences to fixed length.
     */
    public List<Integer> padSequence(List<Integer> sequence, int maxLength) {
        List<Integer> padded = new ArrayList<>(sequence);
        
        // Truncate if too long
        if (padded.size() > maxLength) {
            padded = padded.subList(0, maxLength);
        }
        
        // Pad if too short
        while (padded.size() < maxLength) {
            padded.add(vocab.get(PAD_TOKEN));
        }
        
        return padded;
    }
    
    /**
     * Get vocabulary statistics.
     */
    public TokenizationStats getStats() {
        return new TokenizationStats(vocab.size(), subwordPairs.size(), maxVocabSize, enableSubword);
    }
    
    // Private helper methods
    
    private void initializeSpecialTokens() {
        addToken(PAD_TOKEN);
        addToken(UNK_TOKEN);
        addToken(CLS_TOKEN);
        addToken(SEP_TOKEN);
        addToken(MASK_TOKEN);
    }
    
    private void addToken(String token) {
        if (!vocab.containsKey(token)) {
            int id = vocab.size();
            vocab.put(token, id);
            reverseVocab.put(id, token);
        }
    }
    
    private List<String> basicTokenize(String text) {
        List<String> tokens = new ArrayList<>();
        java.util.regex.Matcher matcher = tokenPattern.matcher(text.toLowerCase());
        
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        
        return tokens;
    }
    
    private List<String> advancedTokenize(String text) {
        // Start with basic tokenization
        List<String> tokens = basicTokenize(text);
        
        // Apply subword decomposition if needed
        if (enableSubword) {
            List<String> result = new ArrayList<>();
            for (String token : tokens) {
                if (!vocab.containsKey(token)) {
                    result.addAll(decomposeSubword(token));
                } else {
                    result.add(token);
                }
            }
            return result;
        }
        
        return tokens;
    }
    
    private void buildSubwordPairs(Map<String, Integer> tokenCounts) {
        // Simplified BPE-like algorithm
        // In practice, this would iteratively merge most frequent character pairs
        
        for (Map.Entry<String, Integer> entry : tokenCounts.entrySet()) {
            String token = entry.getKey();
            if (token.length() > 3 && !vocab.containsKey(token)) {
                // Create subword pairs for common tokens not in vocab
                for (int i = 0; i < token.length() - 1; i++) {
                    String pair = token.substring(i, i + 2);
                    subwordPairs.put(pair, subwordPairs.getOrDefault(pair, 0) + entry.getValue());
                }
            }
        }
        
        // Keep only most frequent subword pairs
        List<Map.Entry<String, Integer>> sortedPairs = new ArrayList<>(subwordPairs.entrySet());
        sortedPairs.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        subwordPairs.clear();
        for (int i = 0; i < Math.min(1000, sortedPairs.size()); i++) {
            subwordPairs.put(sortedPairs.get(i).getKey(), sortedPairs.get(i).getValue());
        }
    }
    
    private List<String> decomposeSubword(String token) {
        if (token.length() <= 2) {
            return Arrays.asList(token);
        }
        
        // Try to break token using subword pairs
        List<String> result = new ArrayList<>();
        int start = 0;
        
        while (start < token.length()) {
            boolean found = false;
            
            // Try longest subwords first
            for (int end = Math.min(token.length(), start + 4); end > start + 1; end--) {
                String candidate = token.substring(start, end);
                if (subwordPairs.containsKey(candidate)) {
                    result.add(candidate);
                    start = end;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                // Add single character
                result.add(token.substring(start, start + 1));
                start++;
            }
        }
        
        return result;
    }
    
    private boolean isSpecialToken(String token) {
        return token.equals(PAD_TOKEN) || token.equals(UNK_TOKEN) || 
               token.equals(CLS_TOKEN) || token.equals(SEP_TOKEN) || 
               token.equals(MASK_TOKEN);
    }
    
    private boolean isPunctuation(String token) {
        return token.matches("[.!?;,\"']");
    }
    
    // Getters
    public Map<String, Integer> getVocab() { return new HashMap<>(vocab); }
    public int getVocabSize() { return vocab.size(); }
    public int getTokenId(String token) { return vocab.getOrDefault(token, vocab.get(UNK_TOKEN)); }
    public String getToken(int id) { return reverseVocab.getOrDefault(id, UNK_TOKEN); }
    
    /**
     * Tokenization statistics container.
     */
    public static class TokenizationStats {
        private final int vocabSize;
        private final int subwordPairs;
        private final int maxVocabSize;
        private final boolean subwordEnabled;
        
        public TokenizationStats(int vocabSize, int subwordPairs, int maxVocabSize, boolean subwordEnabled) {
            this.vocabSize = vocabSize;
            this.subwordPairs = subwordPairs;
            this.maxVocabSize = maxVocabSize;
            this.subwordEnabled = subwordEnabled;
        }
        
        @Override
        public String toString() {
            return String.format(
                "TokenizationStats{vocabSize=%d, subwordPairs=%d, maxVocabSize=%d, subwordEnabled=%s}",
                vocabSize, subwordPairs, maxVocabSize, subwordEnabled
            );
        }
        
        // Getters
        public int getVocabSize() { return vocabSize; }
        public int getSubwordPairs() { return subwordPairs; }
        public int getMaxVocabSize() { return maxVocabSize; }
        public boolean isSubwordEnabled() { return subwordEnabled; }
        public double getVocabUtilization() { return (double) vocabSize / maxVocabSize; }
    }
}
