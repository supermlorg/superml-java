/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.superml.drift;

import java.util.ArrayList;
import java.util.List;

/**
 * EDDM (Early Drift Detection Method) implementation for detecting concept drift
 * based on the distance between classification errors.
 * 
 * Reference: Baena-García, M. et al. "Early Drift Detection Method" (2006)
 */
public class EDDMDriftDetector {
    
    private final DriftConfig config;
    private final List<Integer> errorPositions;
    private double averageDistance;
    private double standardDeviationDistance;
    private double minAverageDistance;
    private double minStandardDeviation;
    private int sampleCount;
    private boolean warningFlag;
    
    // EDDM parameters
    private static final double ALPHA_WARNING = 0.95;
    private static final double ALPHA_DRIFT = 0.90;
    
    public EDDMDriftDetector(DriftConfig config) {
        this.config = config;
        this.errorPositions = new ArrayList<>();
        reset();
    }
    
    /**
     * Process a new prediction and detect drift.
     * @param isError Whether the prediction was incorrect
     * @return EDDM drift detection result
     */
    public EDDMResult detectDrift(boolean isError) {
        sampleCount++;
        
        if (isError) {
            errorPositions.add(sampleCount);
        }
        
        // Need at least 2 errors to calculate distances
        if (errorPositions.size() < 2) {
            return new EDDMResult(false, false, 0.0, 0.0, sampleCount);
        }
        
        // Calculate average distance between consecutive errors
        updateDistanceStatistics();
        
        // Update minimum values (best performance point)
        if (sampleCount > config.getMinSamplesForEDDM() && 
            (averageDistance + 2 * standardDeviationDistance) > 
            (minAverageDistance + 2 * minStandardDeviation)) {
            minAverageDistance = averageDistance;
            minStandardDeviation = standardDeviationDistance;
        }
        
        // Check for warning and drift
        boolean newWarning = false;
        boolean isDrift = false;
        
        if (sampleCount > config.getMinSamplesForEDDM() && minAverageDistance > 0) {
            double warningThreshold = ALPHA_WARNING * (minAverageDistance + 2 * minStandardDeviation);
            double driftThreshold = ALPHA_DRIFT * (minAverageDistance + 2 * minStandardDeviation);
            
            double currentLevel = averageDistance + 2 * standardDeviationDistance;
            
            newWarning = currentLevel < warningThreshold;
            isDrift = currentLevel < driftThreshold;
        }
        
        // Update warning flag
        if (isDrift) {
            // Reset after drift detection
            reset();
        } else {
            warningFlag = newWarning;
        }
        
        return new EDDMResult(isDrift, newWarning, averageDistance, standardDeviationDistance, sampleCount);
    }
    
    /**
     * Update distance statistics between consecutive errors.
     */
    private void updateDistanceStatistics() {
        if (errorPositions.size() < 2) {
            averageDistance = 0.0;
            standardDeviationDistance = 0.0;
            return;
        }
        
        // Calculate distances between consecutive errors
        List<Integer> distances = new ArrayList<>();
        for (int i = 1; i < errorPositions.size(); i++) {
            int distance = errorPositions.get(i) - errorPositions.get(i - 1);
            distances.add(distance);
        }
        
        // Calculate mean distance
        averageDistance = distances.stream().mapToInt(Integer::intValue).average().orElse(0.0);
        
        // Calculate standard deviation
        if (distances.size() > 1) {
            double variance = distances.stream()
                .mapToDouble(d -> Math.pow(d - averageDistance, 2))
                .average().orElse(0.0);
            standardDeviationDistance = Math.sqrt(variance);
        } else {
            standardDeviationDistance = 0.0;
        }
    }
    
    /**
     * Reset EDDM detector state.
     */
    public void reset() {
        errorPositions.clear();
        averageDistance = 0.0;
        standardDeviationDistance = 0.0;
        minAverageDistance = Double.MAX_VALUE;
        minStandardDeviation = Double.MAX_VALUE;
        sampleCount = 0;
        warningFlag = false;
    }
    
    public static class EDDMResult {
        public final boolean isDrift;
        public final boolean isWarning;
        public final double averageDistance;
        public final double standardDeviationDistance;
        public final int sampleCount;
        
        public EDDMResult(boolean isDrift, boolean isWarning, double averageDistance,
                         double standardDeviationDistance, int sampleCount) {
            this.isDrift = isDrift;
            this.isWarning = isWarning;
            this.averageDistance = averageDistance;
            this.standardDeviationDistance = standardDeviationDistance;
            this.sampleCount = sampleCount;
        }
    }
}
