#!/bin/bash

# Script to migrate from com.superml to org.superml

echo "🔄 Migrating packages from com.superml to org.superml..."

# Create new directory structure
mkdir -p src/main/java/org/superml
mkdir -p src/test/java/org/superml

# Move all files from com/superml to org/superml
echo "📁 Moving source files..."
if [ -d "src/main/java/com/superml" ]; then
    cp -r src/main/java/com/superml/* src/main/java/org/superml/
    echo "✅ Source files moved"
fi

if [ -d "src/test/java/com/superml" ]; then
    cp -r src/test/java/com/superml/* src/test/java/org/superml/
    echo "✅ Test files moved"
fi

# Update package declarations in all Java files
echo "📝 Updating package declarations..."
find src/main/java/org/superml -name "*.java" -type f -exec sed -i '' 's/package com\.superml/package org.superml/g' {} +
find src/test/java/org/superml -name "*.java" -type f -exec sed -i '' 's/package com\.superml/package org.superml/g' {} +

# Update import statements in all Java files
echo "📝 Updating import statements..."
find src/main/java/org/superml -name "*.java" -type f -exec sed -i '' 's/import com\.superml/import org.superml/g' {} +
find src/test/java/org/superml -name "*.java" -type f -exec sed -i '' 's/import com\.superml/import org.superml/g' {} +

# Update example files in root /examples directory
echo "📝 Updating example files..."
if [ -d "examples" ]; then
    find examples -name "*.java" -type f -exec sed -i '' 's/import com\.superml/import org.superml/g' {} +
    echo "✅ Example imports updated"
fi

# Clean up old com directory
echo "🧹 Cleaning up old directories..."
rm -rf src/main/java/com
rm -rf src/test/java/com

echo "✅ Migration completed!"
echo ""
echo "📋 Summary:"
echo "- Package changed from com.superml to org.superml"
echo "- All Java files updated with new package declarations"
echo "- All import statements updated"
echo "- Example files updated"
echo ""
echo "🚀 Next steps:"
echo "1. Run: mvn clean compile"
echo "2. Run: mvn test"
echo "3. Verify everything works correctly"
