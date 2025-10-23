# Mini E-Learning Platform

## Overview
This project is a mini e-learning platform designed to provide users with an interactive way to learn through various courses. The platform features a home page that lists available courses, a detailed course page that includes lessons and progress tracking, and functionality to mark courses as completed.

## Features
- **Home Page**: Displays a list of courses with links to their respective detail pages.
- **Course Detail Page**: Shows lessons for the selected course and allows users to track their progress.
- **Completion Tracking**: Users can mark courses as completed, which updates their progress.

## Project Structure
```
mini-elearning-platform
├── src
│   ├── index.html          # Home page of the e-learning platform
│   ├── course.html         # Course detail page
│   ├── css
│   │   └── styles.css      # Styles for the application
│   ├── js
│   │   ├── app.js          # Main JavaScript entry point
│   │   ├── courses.js      # Course data
│   │   └── storage.js      # Storage management for course completion
│   └── components
│       ├── course-card.js   # Component for course cards
│       └── lesson-list.js    # Component for lesson lists
├── package.json            # NPM configuration file
└── README.md               # Project documentation
```

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Open the `src/index.html` file in your web browser to view the home page.
4. Use the navigation to explore courses and lessons.

## How to Use
- Click on a course card on the home page to view its details.
- On the course detail page, you can see the list of lessons and track your progress.
- Mark a course as completed by clicking the designated button.

## Technologies Used
- HTML
- CSS
- JavaScript

## License
This project is open-source and available for modification and distribution.