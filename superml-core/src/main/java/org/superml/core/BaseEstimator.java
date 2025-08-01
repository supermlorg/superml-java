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

package org.superml.core;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Base class for all estimators in SuperML.
 * Provides common functionality similar to sklearn.base.BaseEstimator
 */
public abstract class BaseEstimator implements Estimator, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    protected Map<String, Object> params = new HashMap<>();
    
    @Override
    public Map<String, Object> getParams() {
        return new HashMap<>(params);
    }
    
    @Override
    public Estimator setParams(Map<String, Object> params) {
        this.params.putAll(params);
        return this;
    }
    
    /**
     * Set a single parameter.
     * @param key parameter name
     * @param value parameter value
     * @return this estimator instance
     */
    public BaseEstimator setParam(String key, Object value) {
        this.params.put(key, value);
        return this;
    }
    
    /**
     * Get a single parameter.
     * @param key parameter name
     * @param defaultValue default value if parameter not found
     * @return parameter value
     */
    @SuppressWarnings("unchecked")
    protected <T> T getParam(String key, T defaultValue) {
        return (T) params.getOrDefault(key, defaultValue);
    }
}
