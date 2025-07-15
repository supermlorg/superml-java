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

package org.superml.visualization;

/**
 * Base interface for all SuperML visualizations
 */
public interface Visualization {
    
    /**
     * Render the visualization to console output
     */
    void display();
    
    /**
     * Get the visualization as a formatted string
     * @return String representation of the visualization
     */
    String toString();
    
    /**
     * Set the title for the visualization
     * @param title The title to display
     */
    void setTitle(String title);
    
    /**
     * Get the current title
     * @return The current title
     */
    String getTitle();
}
