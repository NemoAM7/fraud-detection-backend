#!/bin/bash

echo "Modifying project structure without moving existing files..."

# Navigate to the src directory
cd src || exit

# Create necessary directories if they don't exist
mkdir -p components pages styles utils hooks context

# Create necessary component files if they don’t exist
touch components/FraudForm.jsx
touch components/BatchFraudCheck.jsx
touch components/FraudReport.jsx
touch components/FraudStats.jsx

# Create necessary pages if they don’t exist
touch pages/Dashboard.jsx
touch pages/Login.jsx

# Create utility and context files if they don’t exist
touch utils/helpers.js
touch hooks/useAuth.js
touch context/AuthContext.js

# Create a Tailwind CSS file if it doesn’t exist
touch styles/tailwind.css

# Output success message
echo "Project structure updated successfully!"
