const courses = [
    {
        id: 1,
        title: "Introduction to JavaScript",
        description: "Learn the basics of JavaScript, the programming language of the web.",
        lessons: [
            { id: 1, title: "Variables and Data Types", completed: false },
            { id: 2, title: "Functions", completed: false },
            { id: 3, title: "DOM Manipulation", completed: false }
        ]
    },
    {
        id: 2,
        title: "HTML & CSS Fundamentals",
        description: "Understand the structure of web pages using HTML and style them with CSS.",
        lessons: [
            { id: 1, title: "HTML Basics", completed: false },
            { id: 2, title: "CSS Selectors", completed: false },
            { id: 3, title: "Responsive Design", completed: false }
        ]
    },
    {
        id: 3,
        title: "React for Beginners",
        description: "Get started with React, a popular JavaScript library for building user interfaces.",
        lessons: [
            { id: 1, title: "Components and Props", completed: false },
            { id: 2, title: "State and Lifecycle", completed: false },
            { id: 3, title: "Handling Events", completed: false }
        ]
    }
];

export default courses;