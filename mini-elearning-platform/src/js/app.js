// This file contains the main JavaScript entry point for the mini e-learning platform.

document.addEventListener('DOMContentLoaded', () => {
    const courseList = document.getElementById('course-list');
    const courses = getCourses(); // Function to retrieve courses from courses.js

    // Function to render courses on the home page
    function renderCourses() {
        courses.forEach(course => {
            const courseCard = createCourseCard(course); // Function from course-card.js
            courseList.appendChild(courseCard);
        });
    }

    // Event listener for course selection
    courseList.addEventListener('click', (event) => {
        const courseId = event.target.closest('.course-card')?.dataset.id;
        if (courseId) {
            window.location.href = `course.html?id=${courseId}`;
        }
    });

    renderCourses();
});

// Function to get courses (to be implemented in courses.js)
function getCourses() {
    return [
        { id: 1, title: 'JavaScript Basics', description: 'Learn the fundamentals of JavaScript.' },
        { id: 2, title: 'HTML & CSS', description: 'Build beautiful websites with HTML and CSS.' },
        { id: 3, title: 'React for Beginners', description: 'Get started with React and build dynamic web applications.' }
    ];
}

// Function to create a course card (to be implemented in course-card.js)
function createCourseCard(course) {
    const card = document.createElement('div');
    card.className = 'course-card';
    card.dataset.id = course.id;
    card.innerHTML = `
        <h3>${course.title}</h3>
        <p>${course.description}</p>
    `;
    return card;
}