import { 
  Github, 
  Linkedin, 
  Mail, 
  Instagram, 
  Youtube, 
  BookOpen, 
  Database, 
  Brain, 
  Code2, 
  Terminal, 
  Eye, 
  LineChart, 
  Monitor,
  Zap,
  Server
} from 'lucide-react';
import { ProjectCategory, SocialLink } from './types';

// Points to the local image file. 
const profilePic = "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&q=80&w=800";

// Social Links are shared across languages
export const SOCIAL_LINKS: SocialLink[] = [
  { name: "LinkedIn", platform: "LinkedIn", url: "https://www.linkedin.com/in/kubra-ozcan/", icon: Linkedin, color: "hover:text-blue-500" },
  { name: "GitHub", platform: "GitHub", url: "https://github.com/Kubra-Ozcan", icon: Github, color: "hover:text-gray-400" },
  { name: "Medium", platform: "Medium", url: "https://medium.com/@kubra.ozcan_98680", icon: BookOpen, color: "hover:text-white" },
  { name: "Email", platform: "Email", url: `mailto:kubraozcan.business@gmail.com`, icon: Mail, color: "hover:text-orange-400" },
  { name: "Instagram", platform: "Instagram", url: "#", icon: Instagram, color: "hover:text-pink-500" },
  { name: "YouTube", platform: "YouTube", url: "#", icon: Youtube, color: "hover:text-red-600" }
];

/* ---------------- ENGLISH DATA ---------------- */

const PERSONAL_INFO_EN = {
  name: "Kübra ÖZCAN",
  title: "DATA SCIENTIST / AI & COMPUTER ENGINEER",
  email: "kubraozcan.business@gmail.com",
  phone: "Phone number available upon request",
  location: "TÜRKİYE",
  gender: "Female",
  dob: "26/06/1999",
  nationality: "Turkish",
  profileImage: profilePic,
  about: `I graduated in January 2025 from Hasan Kalyoncu University with a Bachelor's degree in Computer Engineering (English, full scholarship). Throughout my academic journey, I gained extensive hands-on experience through both domestic and international internships, research projects, and academic–industry collaborations, especially in the fields of data science, artificial intelligence, and software development.

Earlier, in the summer of 2023, I completed a long-term internship at one of the largest corporate companies in the Southeastern Anatolia Region, where I worked actively on C#, .NET, C# Windows Forms, and ERP systems such as SAP/ABAP. During this internship, I took part in production-ready projects, contributed directly to code development, and collaborated with cross-functional teams in real enterprise environments. During my final year, I was awarded an Erasmus+ scholarship and completed a 4-month international internship at the University of Ljubljana, Faculty of Computer Science and Informatics, where I worked in a Computer Vision Laboratory. In this role, I contributed to computer vision research, dataset preparation, model experimentation, and various Python-based CV pipelines, strengthening my expertise in deep learning and image-processing technologies.

My main technical interests include machine learning, deep learning, computer vision, data analysis, and data manipulation. I have worked extensively on data cleaning, feature engineering, and modeling using techniques such as XGBoost, Random Forest, segmentation models, Bayesian approaches, Decision Trees, and ensemble learning. My project experience includes Python-based data science workflows, where I frequently use libraries such as Scikit-learn, Pandas, NumPy, Selenium, TensorFlow, and Keras. I have also developed end-to-end computer vision applications using OpenCV, MediaPipe, and custom datasets.

I gained exposure to MLOps by building data access pipelines, modeling workflows, and automated processes using tools like JAX, ACL, and API integrations. Additionally, I have strong experience with SQL for data extraction, transformation, and analysis. To present analytical insights effectively, I use Tableau, Power BI, Matplotlib, and Seaborn for clear and impactful visualizations.

Beyond technical work, I have always been an active part of university communities. Throughout my studies, I took on leadership and member roles in several student clubs, including theatre, software communities, and Google Developer Groups (GDG). I also participated in multiple volunteer initiatives, contributing to campus life and community-based projects.

One of my proudest achievements is being selected as one of 2,000 data science scholars among thousands of applicants for the Google Artificial Intelligence and Technology Academy. Through this program, I continue to receive advanced technical training, build AI and data science projects, participate in datathons, and consistently strengthen my expertise.

My strongest motivation lies in uncovering insights from complex data, building intelligent systems, and contributing to decision-support processes that create meaningful real-world impact.`,
  resumeUrl: "#"
};

const TYPEWRITER_TEXTS_EN = [
  "I am AI Engineer",
  "I am Computer Engineer",
  "I am Machine Learning Engineer",
  "I am Software Engineer"
];

const SKILLS_DATA_EN = [
  {
    category: "AI & Machine Learning",
    gradient: "from-purple-500 to-pink-500",
    icon: Brain,
    skills: ["TensorFlow", "Keras", "Scikit-learn", "XGBoost", "OpenCV", "MediaPipe", "YOLO", "Deep Learning", "NLP"]
  },
  {
    category: "Data Science",
    gradient: "from-orange-400 to-red-500",
    icon: Database,
    skills: ["Python", "Pandas", "NumPy", "SQL", "Data Analysis", "Feature Engineering", "Tableau", "Power BI", "Matplotlib"]
  },
  {
    category: "Software Development",
    gradient: "from-blue-400 to-cyan-500",
    icon: Code2,
    skills: ["C#", ".NET", "SAP", "ABAP", "ERP", "Python", "SQL", "Git", "Docker", "REST APIs", "OOP", "Software Architecture", "Agile", "Jira"]
  },
  {
    category: "Other",
    gradient: "from-green-400 to-emerald-500",
    icon: Terminal,
    skills: ["Linux", "Bash", "VS Code", "Research", "Technical Writing", "Public Speaking"]
  }
];

const PROJECT_CATEGORIES_EN: ProjectCategory[] = [
  {
    id: "data-analyze-ml",
    title: "Data Analyze & ML",
    count: 12,
    description: "In-depth data analysis, predictive modeling, and machine learning solutions.",
    gradient: "from-orange-400 to-red-500",
    icon: LineChart,
    emoji: "📊",
    path: "/projects/data-analyze-ml",
    projects: [
      {
        id: 403,
        title: "Emotional / Sentiment Analysis",
        description: "I built an NLP-powered sentiment analysis model that classifies text into positive, negative, or neutral categories using TF-IDF and word embeddings. The project analyzes and interprets emotional tones across textual data.",
        tags: ["NLP", "Machine Learning", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1527689368864-3a821dbccc34?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "TF-IDF", "NLP", "Scikit-learn"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Emotional-Analysis"
      },
      {
        id: 409,
        title: "Renting Price Prediction",
        description: "Developed a machine learning model to predict rental prices using regression algorithms, feature engineering, and data preprocessing techniques. The model demonstrates high performance in estimating rental values.",
        tags: ["Machine Learning", "Regression", "Prediction"],
        imageUrl: "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Scikit-learn", "Pandas"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/RentingPricePrediction"
      },
      {
        id: 410,
        title: "NBA Players Success Prediction",
        description: "Predicts player success in the NBA using Naive Bayes classification. Advanced feature engineering, thorough data cleaning, and model evaluation methods were applied to enhance prediction accuracy.",
        tags: ["Machine Learning", "Naive Bayes", "Sports Analytics"],
        imageUrl: "https://images.unsplash.com/photo-1546519638-68e109498ffc?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Naive Bayes", "Feature Engineering"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/NBA-Players-Success-Prediction--with-Naive-Bayes"
      },
      {
        id: 404,
        title: "Marketing Sales Analysis",
        description: "I performed exploratory data analysis (EDA) on marketing and sales datasets, examining multivariate relationships, customer behavior patterns, and sales trends.",
        tags: ["Data Analysis", "EDA", "Marketing"],
        imageUrl: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Pandas", "Matplotlib", "Seaborn"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 405,
        title: "Penguins Dataset Analysis",
        description: "Using the Penguins dataset, I built linear and multiple linear regression models to analyze relationships between physical measurements and species prediction.",
        tags: ["Data Analysis", "Regression", "Machine Learning"],
        imageUrl: "https://images.unsplash.com/photo-1551187067-169150e12d52?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Linear Regression", "Scikit-learn"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 407,
        title: "Hypothesis Testing with Python",
        description: "I applied statistical hypothesis tests such as t-tests, chi-square tests, and proportion tests to determine whether significant differences exist between various data groups.",
        tags: ["Statistics", "Data Analysis"],
        imageUrl: "https://images.unsplash.com/photo-1543286386-2e659306cd6c?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "SciPy", "Statsmodels"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 408,
        title: "ANOVA Testing",
        description: "This project conducts ANOVA testing to compare multiple groups and assess the impact of categorical variables on numerical outcomes.",
        tags: ["Statistics", "Data Analysis"],
        imageUrl: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "ANOVA", "Statistics"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 411,
        title: "Bank Customer Prediction",
        description: "I built a churn prediction model for banking customers using Naive Bayes. The project includes preprocessing, feature selection, and performance evaluation.",
        tags: ["Machine Learning", "Finance", "Classification"],
        imageUrl: "https://images.unsplash.com/photo-1601597111158-2fceff292cd4?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Naive Bayes", "Scikit-learn"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 412,
        title: "Penguin Cluster Estimation",
        description: "This clustering project utilizes the K-Means algorithm to determine optimal cluster groups using silhouette and inertia metrics.",
        tags: ["Machine Learning", "Clustering", "Unsupervised"],
        imageUrl: "https://images.unsplash.com/photo-1598439210625-5067c578f3f6?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "K-Means", "Scikit-learn"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 413,
        title: "Airline Customer Estimation",
        description: "I developed classification models using Decision Tree, Random Forest, and XGBoost algorithms to analyze airline customer satisfaction.",
        tags: ["Machine Learning", "Classification"],
        imageUrl: "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "XGBoost", "Random Forest"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 414,
        title: "Marketing Promotion Analysis",
        description: "This project evaluates the effectiveness of marketing campaigns through comprehensive EDA, segmentation analysis, and statistical evaluation.",
        tags: ["Data Analysis", "Marketing", "Statistics"],
        imageUrl: "https://images.unsplash.com/photo-1533750516457-a7f992034fec?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Pandas", "Data Viz"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 415,
        title: "Air Quality Analysis",
        description: "I analyzed air quality data using probability distributions (normal, exponential, etc.) and descriptive statistics to assess pollution levels.",
        tags: ["Data Analysis", "Environment", "Statistics"],
        imageUrl: "https://images.unsplash.com/photo-1622345688589-9e67d2645601?auto=format&fit=crop&q=80&w=600",
        category: "Data Analyze & ML Projects",
        technologies: ["Python", "Statistics", "Probability"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      }
    ]
  },
  {
    id: "python-projects",
    title: "Python Projects",
    count: 7,
    description: "Versatile Python applications ranging from automation to data science.",
    gradient: "from-yellow-400 to-orange-500",
    icon: Terminal,
    emoji: "🐍",
    path: "/projects/python-projects",
    projects: [
      {
        id: 501,
        title: "WhatsApp Message Bot",
        description: "Developed an automated WhatsApp messaging bot using Python and Selenium WebDriver. The bot is capable of sending predefined messages to specific contacts, automating repetitive communication tasks.",
        tags: ["Automation", "Selenium", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1611606063065-ee7946f0787a?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "Selenium", "WebDriver"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Whattsap_BotwithPythonSelenium"
      },
      {
        id: 502,
        title: "Simpsons Character Analysis",
        description: "Applies deep learning techniques to classify Simpsons characters from images. Using CNN architectures, extensive preprocessing, and data augmentation, the model successfully identifies characters with high accuracy.",
        tags: ["Deep Learning", "CNN", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1620559029047-e7eb98638d17?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "CNN", "Keras", "TensorFlow"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Simpsons-Character-Analysis-with-Deep-Learning-Python"
      },
      {
        id: 503,
        title: "Air Quality Analysis",
        description: "Analyzed air quality measurements using probability distributions (normal, exponential, Poisson, etc.) and descriptive statistics. The project focuses on understanding the distributional structure of pollution levels.",
        tags: ["Data Analysis", "Statistics", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1611273426728-131c909e735e?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "SciPy", "Statistics"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 504,
        title: "Hypothesis Testing",
        description: "Demonstrates statistical hypothesis testing using Python. Tests such as t-tests, chi-square tests, ANOVA, and proportion tests were applied to evaluate significant differences between groups.",
        tags: ["Statistics", "Data Analysis", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "SciPy", "Statsmodels"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 505,
        title: "ANOVA Testing",
        description: "Performed ANOVA to compare numerical outcomes across multiple groups and determine whether group differences were statistically significant. Includes interpretation of p-values, assumptions, and visual analysis.",
        tags: ["Statistics", "Data Analysis", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1543286386-2e659306cd6c?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "Statsmodels", "Pandas"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 506,
        title: "Web Scraping with Python",
        description: "Focuses on extracting structured information from web pages using Python libraries. Includes scraping product data, text content, and metadata from multiple websites, followed by data cleaning.",
        tags: ["Web Scraping", "Python", "Data Mining"],
        imageUrl: "https://images.unsplash.com/photo-1558494949-ef526b0042a0?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "BeautifulSoup", "Selenium"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 507,
        title: "Clothing Images Dataset",
        description: "Created a custom dataset by scraping clothing images from various shopping websites using Selenium. Automates navigation, image extraction, and dataset structuring for computer vision tasks.",
        tags: ["Web Scraping", "Dataset", "Python"],
        imageUrl: "https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?auto=format&fit=crop&q=80&w=600",
        category: "Python Projects",
        technologies: ["Python", "Selenium", "Computer Vision"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Web-Scrapping-with-Python-_-Clothes-"
      }
    ]
  },
  {
    id: "computer-vision",
    title: "Computer Vision Projects",
    count: 7,
    description: "Advanced image processing, deep learning, and real-time vision systems.",
    gradient: "from-purple-500 to-pink-500",
    icon: Eye,
    emoji: "👁️",
    path: "/projects/computer-vision",
    projects: [
      {
        id: 801,
        title: "Face Detection with Haar Cascades",
        description: "In this project, I implemented a face detection system using the Haar Cascade Classifier in Python and OpenCV. The model detects human faces in real time by leveraging classical computer vision techniques. I also published a detailed Medium article explaining the full implementation, preprocessing steps, and underlying algorithmic logic.",
        tags: ["Computer Vision", "OpenCV", "Haar Cascade"],
        imageUrl: "https://images.unsplash.com/photo-1555685812-4b943f3db9f0?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "OpenCV", "Haar Cascade"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/-Face-Detection-with-Haar-Cascades-Computer-Vision",
        articles: [
            { label: "Read in English", url: "https://medium.com/@kubra.ozcan_98680/face-recognition-with-haar-cascade-classifier-in-python-6439da0d23e3", lang: 'en' },
            { label: "Türkçe Oku", url: "https://medium.com/@kubra.ozcan_98680/python-da-haar-cascade-s%C4%B1n%C4%B1fland%C4%B1r%C4%B1c%C4%B1-ile-y%C3%BCz-tan%C4%B1ma-4f0e0cb13f2c", lang: 'tr' }
        ]
      },
      {
        id: 802,
        title: "Face Recognition with OpenCV",
        description: "This project focuses on building a face recognition system using OpenCV and deep learning–based facial embeddings. The application identifies individuals by comparing extracted facial features with a trained dataset. It highlights preprocessing, face encoding, model training, and real-time video inference.",
        tags: ["Computer Vision", "Deep Learning", "OpenCV", "Face Recognition"],
        imageUrl: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "OpenCV", "Deep Learning"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Face-Recognition-with-OpenCV-s---Computer-Vision-Deep-Learning"
      },
      {
        id: 803,
        title: "Edge Detection with Python",
        description: "I developed an edge detection pipeline using OpenCV to analyze object boundaries within images. Techniques such as Canny Edge Detection and gradient-based filters were applied to extract meaningful structural details from visual data.",
        tags: ["Computer Vision", "OpenCV", "Image Processing"],
        imageUrl: "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "OpenCV", "Canny Edge"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Edge-Detection-with-Python-"
      },
      {
        id: 2,
        title: "Hand Tracking and Gesture Control",
        description: "This project implements real-time hand tracking and gesture recognition using MediaPipe and OpenCV. The system detects hand landmarks from a live camera feed and maps gestures to specific control actions, enabling intuitive human–computer interaction.",
        tags: ["Computer Vision", "MediaPipe", "Real-time", "HCI"],
        imageUrl: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "OpenCV", "MediaPipe"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Hand-Tracking-and-Gesture-Control-from-Camera-Using-Mediapipe-and-OpenCV-in-Python"
      },
      {
        id: 804,
        title: "Vehicle Tracking and Counting",
        description: "In this computer vision project, I built a vehicle detection, tracking, and counting system using the YOLOv8 object detection model. The application processes video streams to detect moving vehicles, track their trajectories, and count them based on lane crossings.",
        tags: ["Computer Vision", "YOLOv8", "Object Detection", "Tracking"],
        imageUrl: "https://images.unsplash.com/photo-1565514020176-db936c646002?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "YOLOv8", "OpenCV"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Track-and-Count-Vehicles-with-yolov8"
      },
      {
        id: 401,
        title: "Flower Recognition",
        description: "Developed a deep-learning-based image classification model capable of identifying different flower species from images using CNN architectures.",
        tags: ["Deep Learning", "CNN", "Classification"],
        imageUrl: "https://images.unsplash.com/photo-1490750967868-bcdf92dd8364?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "CNN", "Keras", "TensorFlow"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Flower_Recognition_Deep_Learning_Project"
      },
      {
        id: 402,
        title: "Simpsons Character Recognition",
        description: "Focuses on recognizing Simpsons characters through image classification. I developed a deep learning model using CNNs and comprehensive preprocessing.",
        tags: ["Deep Learning", "CNN", "Classification"],
        imageUrl: "https://images.unsplash.com/photo-1580130601275-c9f0c2a4dd85?auto=format&fit=crop&q=80&w=600",
        category: "Computer Vision Projects",
        technologies: ["Python", "CNN", "Deep Learning"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Simpsons-Character-Analysis-with-Deep-Learning-Python"
      }
    ]
  },
  {
    id: "ai-projects",
    title: "AI & Automation Projects",
    count: 2,
    description: "Cutting-edge artificial intelligence applications and automated systems.",
    gradient: "from-blue-400 to-cyan-500",
    icon: Brain,
    emoji: "🧠",
    path: "/projects/ai-projects",
    projects: [
      {
        id: 601,
        title: "AI Assistant App",
        description: "This project is an interactive AI assistant application built using LangChain and Streamlit. It allows users to ask questions, receive contextual and intelligent responses, and explore personalized roadmap suggestions—powered by large language models. The app integrates prompt chaining, dynamic UI components, and real-time interaction features to deliver a seamless conversational experience.",
        tags: ["AI", "LangChain", "Streamlit", "LLM"],
        imageUrl: "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=600",
        category: "AI Projects",
        technologies: ["Python", "LangChain", "Streamlit", "OpenAI"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/AI-App-with-LangChain-and-Streamlit"
      },
       {
        id: 406,
        title: "Diabetes Risk Prediction App",
        description: "This application predicts an individual's diabetes risk using multiple machine learning classification models. The project includes data preprocessing, feature analysis, model training, and prediction logic wrapped in a user-friendly interface. It enables users to input medical parameters and immediately view their risk level based on trained ML models.",
        tags: ["Healthcare", "Prediction", "Classification", "App"],
        imageUrl: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&q=80&w=600",
        category: "AI Projects",
        technologies: ["Python", "Scikit-learn", "Pandas", "Streamlit"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/Diabetes-Risk-Prediction-App"
      }
    ]
  },
  {
    id: "end-to-end",
    title: "End-to-End Projects",
    count: 6,
    description: "Full-stack and complete lifecycle projects from conception to deployment.",
    gradient: "from-green-400 to-emerald-500",
    icon: Monitor,
    emoji: "🖥️",
    path: "/projects/end-to-end",
    projects: [
      {
        id: 202,
        title: "Portfolio Website",
        description: "Modern portfolio website built with React and Tailwind CSS.",
        tags: ["Web Dev", "Frontend"],
        imageUrl: "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?auto=format&fit=crop&q=80&w=600",
        category: "End-to-End Projects",
        technologies: ["React", "TypeScript", "Tailwind CSS", "Framer Motion"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan"
      },
      {
        id: 701,
        title: "Donation Website",
        description: "This end-to-end web application is a donation platform where users can contribute clothes, books, and other items. I developed both the frontend and backend using HTML, CSS, JavaScript, and PHP. The system includes user-friendly forms, database operations via XAMPP, item submission workflows, and a fully functional donation management interface.",
        tags: ["Web Dev", "PHP", "Full Stack"],
        imageUrl: "https://images.unsplash.com/photo-1532629345422-7515f3d16bb6?auto=format&fit=crop&q=80&w=600",
        category: "End-to-End Projects",
        technologies: ["HTML", "CSS", "PHP", "JavaScript", "XAMPP"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/DonationWebProject_With_Html_CSS_PHP_JS"
      },
      {
        id: 702,
        title: "Traversal Core Project",
        description: "This project was developed as part of Murat Yücedağ’s comprehensive C# .NET course. It is a full-stack MVC-based travel management system built using .NET Core. The application includes layered architecture, user authentication, admin panels, dynamic content management, and database integration—demonstrating real-world enterprise-level software design.",
        tags: [".NET Core", "MVC", "Full Stack"],
        imageUrl: "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?auto=format&fit=crop&q=80&w=600",
        category: "End-to-End Projects",
        technologies: ["C#", ".NET Core", "MVC", "SQL"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/TraversalCoreProje"
      },
      {
        id: 703,
        title: "Car Rental Automation",
        description: "I built a complete car rental automation system using C# and .NET technologies. The project includes vehicle management, customer registration, rental transactions, payment structure, and reporting features. It demonstrates practical usage of OOP principles, CRUD operations, and database handling in .NET environments.",
        tags: ["Desktop App", "Automation", "C#"],
        imageUrl: "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&q=80&w=600",
        category: "End-to-End Projects",
        technologies: ["C#", ".NET", "OOP", "SQL"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/CarRentalProject"
      },
      {
        id: 704,
        title: "Customer Sales Automation",
        description: "This project is a desktop-based sales automation system developed with C# Windows Forms and Entity Framework. It includes modules for customer management, product tracking, sales processing, and real-time data operations. Entity Framework was used for database modeling, relational mapping, and efficient data transactions.",
        tags: ["Desktop App", "WinForms", "Database"],
        imageUrl: "https://images.unsplash.com/photo-1556742049-0cfed4f7a07d?auto=format&fit=crop&q=80&w=600",
        category: "End-to-End Projects",
        technologies: ["C#", "WinForms", "Entity Framework", "SQL"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/EntityFramework"
      },
      {
        id: 705,
        title: "Car Rental C# Form Website",
        description: "This is a web-based car rental platform built using C# and .NET technologies. It includes functionalities for vehicle listing, customer booking, rental tracking, and administrative control panels. The project reflects a complete end-to-end system with backend logic, database operations, and a dynamic user interface.",
        tags: ["Web Dev", ".NET", "Full Stack"],
        imageUrl: "https://images.unsplash.com/photo-1485291571150-772bcfc10da5?auto=format&fit=crop&q=80&w=600",
        category: "End-to-End Projects",
        technologies: ["C#", ".NET", "ASP.NET", "SQL"],
        link: "#",
        github: "https://github.com/Kubra-Ozcan/ArabaKiralama"
      }
    ]
  },
  {
    id: "other-projects",
    title: "Other Projects",
    count: 0,
    description: "Various other technical projects and experiments.",
    gradient: "from-gray-500 to-slate-500",
    icon: Zap,
    emoji: "🌐",
    path: "/projects/other-projects",
    projects: []
  }
];

const RESUME_DATA_EN = {
  experience: [
    {
      role: "Core Team Member",
      company: "Kaggle Türkiye Community",
      period: "June 2025 - Present",
      description: "Organizing AI and data science events in Turkey; managing sponsorships and partnerships to align with global trends and foster local engagement."
    },
    {
      role: "Data Science Scholar",
      company: "Google AI & Technology Academy",
      period: "November 2024 - Present",
      description: "Gained valuable project management experience while participating in hackathons, entrepreneurship training, and data science competitions, building a strong network of professionals in the data science community."
    },
    {
      role: "Contributor - Trainer (Part Time)",
      company: "Outlier",
      period: "November 2024 - Present",
      description: "Worked part-time on training AI models by optimizing and experimenting with language models at Outlier."
    },
    {
      role: "Volunteer",
      company: "T3 Foundation",
      period: "April 2024 - Present",
      description: "Volunteering in Machine Learning, Computer Vision, and Deep Learning initiatives."
    },
    {
      role: "Erasmus+ Exchange Internship",
      company: "University of Ljubljana Computer Vision Laboratory",
      period: "February 2024 - August 2024",
      description: "Conducted research in the field of machine learning, computer vision, and deep learning. Worked on real-world computer vision projects, utilizing advanced algorithms and neural networks to analyze visual data and extract meaningful insights."
    },
    {
      role: "Intern Engineer",
      company: "SANKO Holding A.Ş (SANShine)",
      period: "July 2023 - October 2023",
      description: "Assisted in ERP system integration and management, including working with SAP, ABAP, SQL, C#, and .NET Form technologies. Contributed to the development and optimization of internal tools for the company's operations."
    },
    {
      role: "Event Organizer",
      company: "HKU GDSC Core Team Member",
      period: "September 2023 - May 2024",
      description: "Organized tech events, managed sponsorships, and facilitated partnerships to enhance community engagement and provide valuable opportunities for students."
    },
    {
      role: "Project Team Coordinator",
      company: "2209 TUBITAK",
      period: "October 2023 - April 2024",
      description: "University Students Research Projects Support Program - VR Simulation Project for Police Schools (Project Approved). Web Programming (PHP, CSS, HTML, Javascript), Game Developing (Unity)."
    },
    {
      role: "One-to-One Training Volunteer",
      company: "Bamboo Training Platform",
      period: "October 2022 - December 2024",
      description: "Provided voluntary mathematics tutoring to a high school student."
    }
  ],
  education: [
    {
      school: "Hasan Kalyoncu University",
      degree: "Computer Engineering (English) - Bachelor's Degree",
      period: "2019-2025",
      detail: "Full Scholarship"
    },
    {
      school: "University of Ljubljana",
      degree: "Erasmus+ Exchange",
      period: "2024",
      detail: "Computer Vision Laboratory"
    },
    {
      school: "Gaziantep Anatolian High School",
      degree: "High School",
      period: "",
      detail: "(TÜRKİYE)"
    }
  ],
  skills: {
    technical: [
      "Python", "SQL", "Machine Learning", "Deep Learning", "Computer Vision", 
      "Data Science", "Scikit-learn", "Pandas", "NumPy", "TensorFlow", "Keras", "OpenCV", 
      "Tableau", "Power BI", "Git", "C#", ".NET", "SAP/ABAP", "MediaPipe", "JAX", "Selenium"
    ],
    soft: [
      "Team Work", "Integrity", "Leadership", "Analytical Problem-Solving"
    ]
  },
  languages: [
    { name: "English", level: "IELTS: B2" },
    { name: "German", level: "A2" },
    { name: "Turkish", level: "Native" }
  ],
  certificates: [
    "Google AI Essentials (Google)",
    "IELTS Certificate (Level: B2)",
    "Advanced Data Analytics (Coursera - Google)",
    "Miuul Machine Learning",
    "Python for Machine Learning (AI Business School)",
    "Google Project Management (Coursera - Google)",
    "Data Analysis Bootcamp (Global AI Hub)",
    "E-Commerce Masterclass (The Dot Academy)"
  ],
  hobbies: [
    "Doing Yoga", "Playing Piano", "Amateur Theatre Acting", "Playing Chess", "Drawing"
  ],
  references: [
    {
      name: "Prof. Dr. Muhammet Fatih HASOĞLU",
      title: "Head of Computer Engineering Department at Hasan Kalyoncu University",
      contact: "References' contact information is available upon request"
    },
    {
      name: "Ersan TAN (Sanko Holding)",
      title: "Head of Logistics (HR/MM/EWM)",
      contact: "References' contact information is available upon request"
    }
  ]
};

/* ---------------- TURKISH DATA ---------------- */

const PERSONAL_INFO_TR = {
  ...PERSONAL_INFO_EN,
  title: "VERİ BİLİMCİSİ / YAPAY ZEKA VE BİLGİSAYAR MÜHENDİSİ",
  about: `Ocak 2025'te Hasan Kalyoncu Üniversitesi Bilgisayar Mühendisliği (İngilizce, tam burslu) bölümünden mezun oldum. Akademik yolculuğum boyunca, özellikle veri bilimi, yapay zeka ve yazılım geliştirme alanlarında yurtiçi ve yurtdışı stajlar, araştırma projeleri ve akademik-sektörel iş birlikleri yoluyla kapsamlı deneyimler kazandım.

Daha önce, 2023 yazında Güneydoğu Anadolu Bölgesi'nin en büyük kurumsal şirketlerinden birinde uzun dönem stajımı tamamladım. Burada C#, .NET, C# Windows Formları ve SAP/ABAP gibi ERP sistemleri üzerinde aktif olarak çalıştım. Bu staj süresince üretime hazır projelerde yer aldım, doğrudan kod geliştirmeye katkıda bulundum ve gerçek kurumsal ortamlarda ekipler arası iş birliği yaptım. Son yılımda Erasmus+ bursu kazanarak Slovenya'daki Ljubljana Üniversitesi, Bilgisayar ve Enformatik Fakültesi'nde 4 aylık uluslararası bir staj tamamladım. Buradaki Bilgisayar Görüsü Laboratuvarı'nda (Computer Vision Laboratory) çalıştım. Bu rolde bilgisayarlı görü araştırmalarına, veri seti hazırlığına, model deneylerine ve çeşitli Python tabanlı görüntü işleme süreçlerine katkıda bulunarak derin öğrenme ve görüntü işleme teknolojilerindeki uzmanlığımı güçlendirdim.

Başlıca teknik ilgi alanlarım arasında makine öğrenimi, derin öğrenme, bilgisayarlı görü, veri analizi ve veri manipülasyonu yer almaktadır. Veri temizleme, özellik mühendisliği ve XGBoost, Random Forest, segmentasyon modelleri, Bayes yaklaşımları, Karar Ağaçları ve topluluk öğrenimi gibi teknikleri kullanarak modelleme konularında kapsamlı çalışmalar yaptım. Proje deneyimlerim arasında Scikit-learn, Pandas, NumPy, Selenium, TensorFlow ve Keras gibi kütüphaneleri sıkça kullandığım Python tabanlı veri bilimi iş akışları bulunmaktadır. Ayrıca OpenCV, MediaPipe ve özel veri setleri kullanarak uçtan uca bilgisayarlı görü uygulamaları geliştirdim.

JAX, ACL ve API entegrasyonları gibi araçları kullanarak veri erişim hatları, modelleme iş akışları ve otomatik süreçler oluşturarak MLOps konusunda deneyim kazandım. Ayrıca, veri çıkarma, dönüştürme ve analiz süreçlerinde SQL konusunda güçlü bir deneyime sahibim. Analitik içgörüleri etkili bir şekilde sunmak için Tableau, Power BI, Matplotlib ve Seaborn gibi araçları kullanarak net ve etkileyici görselleştirmeler yapıyorum.

Teknik çalışmalarımın ötesinde, üniversite topluluklarının her zaman aktif bir parçası oldum. Öğrenimim boyunca tiyatro, yazılım toplulukları ve Google Developer Groups (GDG) dahil olmak üzere çeşitli öğrenci kulüplerinde liderlik ve üyelik rolleri üstlendim. Ayrıca kampüs yaşamına ve toplum temelli projelere katkıda bulunarak birçok gönüllü girişimde yer aldım.

En gurur duyduğum başarılarımdan biri, binlerce başvuru arasından Google Yapay Zeka ve Teknoloji Akademisi için seçilen 2.000 veri bilimi bursiyerinden biri olmaktır. Bu program sayesinde ileri düzey teknik eğitimler almaya, yapay zeka ve veri bilimi projeleri geliştirmeye, datathonlara katılmaya ve uzmanlığımı sürekli güçlendirmeye devam ediyorum.

En güçlü motivasyonum, karmaşık verilerden içgörüler ortaya çıkarmak, akıllı sistemler inşa etmek ve anlamlı, gerçek dünya etkileri yaratan karar destek süreçlerine katkıda bulunmaktır.`,
};

const TYPEWRITER_TEXTS_TR = [
  "Ben Yapay Zeka Mühendisiyim",
  "Ben Bilgisayar Mühendisiyim",
  "Ben Makine Öğrenmesi Mühendisiyim",
  "Ben Yazılım Mühendisiyim"
];

const SKILLS_DATA_TR = SKILLS_DATA_EN;

const PROJECT_CATEGORIES_TR: ProjectCategory[] = JSON.parse(JSON.stringify(PROJECT_CATEGORIES_EN));
PROJECT_CATEGORIES_TR.forEach(cat => {
    if(cat.id === "data-analyze-ml") {
        cat.title = "Veri Analizi & ML";
        cat.description = "Derinlemesine veri analizi, tahmine dayalı modelleme ve makine öğrenimi çözümleri.";
    } else if (cat.id === "python-projects") {
        cat.title = "Python Projeleri";
        cat.description = "Otomasyondan veri bilimine kadar çok yönlü Python uygulamaları.";
    } else if (cat.id === "computer-vision") {
        cat.title = "Bilgisayarlı Görü";
        cat.description = "Gelişmiş görüntü işleme, derin öğrenme ve gerçek zamanlı görü sistemleri.";
    } else if (cat.id === "ai-projects") {
        cat.title = "Yapay Zeka Projeleri";
        cat.description = "Son teknoloji yapay zeka uygulamaları ve otomatik sistemler.";
    } else if (cat.id === "end-to-end") {
        cat.title = "Uçtan Uca Projeler";
        cat.description = "Fikirden dağıtıma kadar tam kapsamlı ve tam döngü projeler.";
    } else if (cat.id === "other-projects") {
        cat.title = "Diğer Projeler";
        cat.description = "Çeşitli diğer teknik projeler ve deneyler.";
    }
});

const RESUME_DATA_TR = {
  experience: [
    {
      role: "Çekirdek Ekip Üyesi",
      company: "Kaggle Türkiye Topluluğu",
      period: "Haziran 2025 - Günümüz",
      description: "Türkiye'de yapay zeka ve veri bilimi etkinlikleri düzenlemek; küresel trendlerle uyum sağlamak ve yerel katılımı teşvik etmek için sponsorlukları ve ortaklıkları yönetmek."
    },
    {
      role: "Veri Bilimi Bursiyeri",
      company: "Google Yapay Zeka ve Teknoloji Akademisi",
      period: "Kasım 2024 - Günümüz",
      description: "Hackathonlara, girişimcilik eğitimlerine ve veri bilimi yarışmalarına katılarak değerli proje yönetimi deneyimi kazandım ve veri bilimi topluluğunda güçlü bir profesyonel ağ oluşturdum."
    },
    {
      role: "Katılımcı - Eğitmen (Yarı Zamanlı)",
      company: "Outlier",
      period: "Kasım 2024 - Günümüz",
      description: "Outlier'da dil modellerini optimize ederek ve deneyler yaparak yapay zeka modellerini eğitmek üzerine yarı zamanlı çalıştım."
    },
    {
      role: "Gönüllü",
      company: "T3 Vakfı",
      period: "Nisan 2024 - Günümüz",
      description: "Makine Öğrenimi, Bilgisayarlı Görü ve Derin Öğrenme girişimlerinde gönüllü olarak yer alıyorum."
    },
    {
      role: "Erasmus+ Değişim Stajyeri",
      company: "Ljubljana Üniversitesi Bilgisayarlı Görü Laboratuvarı",
      period: "Şubat 2024 - Ağustos 2024",
      description: "Makine öğrenimi, bilgisayarlı görü ve derin öğrenme alanlarında araştırmalar yürüttüm. Görsel verileri analiz etmek ve anlamlı içgörüler elde etmek için gelişmiş algoritmalar ve sinir ağları kullanarak gerçek dünya bilgisayarlı görü projelerinde çalıştım."
    },
    {
      role: "Stajyer Mühendis",
      company: "SANKO Holding A.Ş (SANShine)",
      period: "Temmuz 2023 - Ekim 2023",
      description: "SAP, ABAP, SQL, C# ve .NET Form teknolojileriyle çalışarak ERP sistem entegrasyonu ve yönetimine yardımcı oldum. Şirket operasyonları için iç araçların geliştirilmesine ve optimizasyonuna katkıda bulundum."
    },
    {
      role: "Etkinlik Organizatörü",
      company: "HKU GDSC Çekirdek Ekip Üyesi",
      period: "Eylül 2023 - Mayıs 2024",
      description: "Teknoloji etkinlikleri düzenledim, sponsorlukları yönettim ve topluluk katılımını artırmak ve öğrencilere değerli fırsatlar sunmak için ortaklıklar kurdum."
    },
    {
      role: "Proje Ekip Koordinatörü",
      company: "2209 TUBITAK",
      period: "Ekim 2023 - Nisan 2024",
      description: "Üniversite Öğrencileri Araştırma Projeleri Destek Programı - Polis Okulları için VR Simülasyon Projesi (Proje Onaylandı). Web Programlama (PHP, CSS, HTML, Javascript), Oyun Geliştirme (Unity)."
    },
    {
      role: "Birebir Eğitim Gönüllüsü",
      company: "Bamboo Eğitim Platformu",
      period: "Ekim 2022 - Aralık 2024",
      description: "Bir lise öğrencisine gönüllü matematik özel dersi verdim."
    }
  ],
  education: [
    {
      school: "Hasan Kalyoncu Üniversitesi",
      degree: "Bilgisayar Mühendisliği (İngilizce) - Lisans Derecesi",
      period: "2019-2025",
      detail: "Tam Burslu"
    },
    {
      school: "Ljubljana Üniversitesi",
      degree: "Erasmus+ Değişimi",
      period: "2024",
      detail: "Bilgisayarlı Görü Laboratuvarı"
    },
    {
      school: "Gaziantep Anadolu Lisesi",
      degree: "Lise",
      period: "",
      detail: "(TÜRKİYE)"
    }
  ],
  skills: {
    technical: SKILLS_DATA_EN[0].skills.concat(SKILLS_DATA_EN[1].skills, SKILLS_DATA_EN[2].skills, SKILLS_DATA_EN[3].skills), 
    soft: [
      "Takım Çalışması", "Dürüstlük", "Liderlik", "Analitik Problem Çözme"
    ]
  },
  skillsDisplay: {
    technical: [
      "Python", "SQL", "Machine Learning", "Deep Learning", "Computer Vision", 
      "Data Science", "Scikit-learn", "Pandas", "NumPy", "TensorFlow", "Keras", "OpenCV", 
      "Tableau", "Power BI", "Git", "C#", ".NET", "SAP/ABAP", "MediaPipe", "JAX", "Selenium"
    ],
    soft: [
      "Takım Çalışması", "Dürüstlük", "Liderlik", "Analitik Problem Çözme"
    ]
  },
  languages: [
    { name: "İngilizce", level: "IELTS: B2" },
    { name: "Almanca", level: "A2" },
    { name: "Türkçe", level: "Anadil" }
  ],
  certificates: [
    "Google AI Essentials (Google)",
    "IELTS Sertifikası (Seviye: B2)",
    "İleri Veri Analitiği (Coursera - Google)",
    "Miuul Makine Öğrenmesi",
    "Makine Öğrenmesi için Python (AI Business School)",
    "Google Proje Yönetimi (Coursera - Google)",
    "Veri Analizi Bootcamp (Global AI Hub)",
    "E-Ticaret Masterclass (The Dot Academy)"
  ],
  hobbies: [
    "Yoga Yapmak", "Piyano Çalmak", "Amatör Tiyatro Oyunculuğu", "Satranç Oynamak", "Çizim Yapmak"
  ],
  references: [
    {
      name: "Prof. Dr. Muhammet Fatih HASOĞLU",
      title: "Hasan Kalyoncu Üniversitesi Bilgisayar Mühendisliği Bölüm Başkanı",
      contact: "Referansların iletişim bilgileri talep üzerine sunulur"
    },
    {
      name: "Ersan TAN (Sanko Holding)",
      title: "Lojistik Müdürü (İK/MM/EWM)",
      contact: "Referansların iletişim bilgileri talep üzerine sunulur"
    }
  ]
};

const SKILLS_DATA_TR_RESUME = {
    technical: SKILLS_DATA_EN.flatMap(s => s.skills),
    soft: RESUME_DATA_TR.skills.soft
};
RESUME_DATA_TR.skills = SKILLS_DATA_TR_RESUME as any;

/* ---------------- GERMAN DATA (DE) ---------------- */

const PERSONAL_INFO_DE = {
  ...PERSONAL_INFO_EN,
  title: "DATA SCIENTIST / KI- & COMPUTER-INGENIEURIN",
  about: `Ich habe im Januar 2025 meinen Bachelor-Abschluss in Computertechnik (Englisch, Vollstipendium) an der Hasan Kalyoncu Universität gemacht. Während meiner akademischen Laufbahn sammelte ich umfangreiche praktische Erfahrungen durch Praktika im In- und Ausland, Forschungsprojekte und Kooperationen zwischen Industrie und Hochschule, insbesondere in den Bereichen Data Science, künstliche Intelligenz und Softwareentwicklung.

Zuvor, im Sommer 2023, absolvierte ich ein Langzeitpraktikum bei einem der größten Unternehmen in der Region Südostanatolien, wo ich aktiv mit C#, .NET, C# Windows Forms und ERP-Systemen wie SAP/ABAP arbeitete. In dieser Zeit war ich an produktionsreifen Projekten beteiligt, trug direkt zur Code-Entwicklung bei und arbeitete mit funktionsübergreifenden Teams in realen Unternehmensumgebungen zusammen. In meinem letzten Studienjahr erhielt ich ein Erasmus+-Stipendium und absolvierte ein 4-monatiges internationales Praktikum an der Universität Ljubljana, Fakultät für Computer- und Informationswissenschaft, wo ich im Computer Vision Laboratory arbeitete. Dort trug ich zur Forschung im Bereich Computer Vision, zur Vorbereitung von Datensätzen, zu Modellexperimenten und verschiedenen Python-basierten CV-Pipelines bei und stärkte meine Expertise in Deep Learning und Bildverarbeitungstechnologien.

Zu meinen technischen Interessen gehören maschinelles Lernen, Deep Learning, Computer Vision, Datenanalyse und Datenmanipulation. Ich habe umfangreiche Arbeiten in den Bereichen Datenbereinigung, Feature Engineering und Modellierung mit Techniken wie XGBoost, Random Forest, Segmentierungsmodellen, Bayes-Ansätzen, Entscheidungsbäumen und Ensemble-Learning durchgeführt. Meine Projekterfahrung umfasst Python-basierte Data-Science-Workflows, bei denen ich häufig Bibliotheken wie Scikit-learn, Pandas, NumPy, Selenium, TensorFlow und Keras verwende. Zudem habe ich End-to-End-Computer-Vision-Anwendungen mit OpenCV, MediaPipe und benutzerdefinierten Datensätzen entwickelt.`,
};

const TYPEWRITER_TEXTS_DE = [
  "Ich bin KI-Ingenieurin",
  "Ich bin Computer-Ingenieurin",
  "Ich bin Machine Learning Ingenieurin",
  "Ich bin Software-Ingenieurin"
];

const SKILLS_DATA_DE = SKILLS_DATA_EN;

const PROJECT_CATEGORIES_DE: ProjectCategory[] = JSON.parse(JSON.stringify(PROJECT_CATEGORIES_EN));
PROJECT_CATEGORIES_DE.forEach(cat => {
    if(cat.id === "data-analyze-ml") {
        cat.title = "Datenanalyse & ML";
        cat.description = "Tiefgehende Datenanalyse, prädiktive Modellierung und Lösungen für maschinelles Lernen.";
    } else if (cat.id === "python-projects") {
        cat.title = "Python-Projekte";
        cat.description = "Vielseitige Python-Anwendungen von Automatisierung bis Data Science.";
    } else if (cat.id === "computer-vision") {
        cat.title = "Computer Vision";
        cat.description = "Fortschrittliche Bildverarbeitung, Deep Learning und Echtzeit-Vision-Systeme.";
    } else if (cat.id === "ai-projects") {
        cat.title = "KI-Projekte";
        cat.description = "Hochmoderne Anwendungen künstlicher Intelligenz und automatisierte Systeme.";
    } else if (cat.id === "end-to-end") {
        cat.title = "End-to-End-Projekte";
        cat.description = "Full-Stack- und Komplettzyklus-Projekte von der Konzeption bis zur Bereitstellung.";
    } else if (cat.id === "other-projects") {
        cat.title = "Andere Projekte";
        cat.description = "Verschiedene andere technische Projekte und Experimente.";
    }
});

const RESUME_DATA_DE = {
  experience: [
    {
      role: "Kernteam-Mitglied",
      company: "Kaggle Türkiye Community",
      period: "Juni 2025 - Heute",
      description: "Organisation von KI- und Data-Science-Events in der Türkei; Verwaltung von Sponsoring und Partnerschaften."
    },
    {
      role: "Data Science Stipendiatin",
      company: "Google AI & Technology Academy",
      period: "November 2024 - Heute",
      description: "Sammeln wertvoller Projektmanagement-Erfahrungen durch Teilnahme an Hackathons, Unternehmerschulungen und Data-Science-Wettbewerben."
    },
    {
      role: "Mitwirkende - Trainerin (Teilzeit)",
      company: "Outlier",
      period: "November 2024 - Heute",
      description: "Teilzeitarbeit am Training von KI-Modellen durch Optimierung und Experimentieren mit Sprachmodellen."
    },
    {
      role: "Freiwillige",
      company: "T3 Foundation",
      period: "April 2024 - Heute",
      description: "Freiwilligenarbeit in Initiativen für Maschinelles Lernen, Computer Vision und Deep Learning."
    },
    {
      role: "Erasmus+ Austauschpraktikantin",
      company: "Universität Ljubljana Computer Vision Labor",
      period: "Februar 2024 - August 2024",
      description: "Forschung im Bereich maschinelles Lernen, Computer Vision und Deep Learning. Arbeit an realen Computer-Vision-Projekten."
    },
    {
      role: "Ingenieurspraktikantin",
      company: "SANKO Holding A.Ş (SANShine)",
      period: "Juli 2023 - Oktober 2023",
      description: "Unterstützung bei der Integration und Verwaltung von ERP-Systemen (SAP, ABAP, SQL, C#, .NET)."
    },
    {
      role: "Event-Organisatorin",
      company: "HKU GDSC Kernteam-Mitglied",
      period: "September 2023 - Mai 2024",
      description: "Organisation von Technologie-Events, Verwaltung von Sponsoring und Förderung von Partnerschaften."
    },
    {
      role: "Projektteam-Koordinatorin",
      company: "2209 TUBITAK",
      period: "Oktober 2023 - April 2024",
      description: "Forschungsprojekt-Unterstützungsprogramm für Universitätsstudenten - VR-Simulationsprojekt."
    },
    {
      role: "Eins-zu-Eins-Trainingsfreiwillige",
      company: "Bamboo Training Platform",
      period: "Oktober 2022 - Dezember 2024",
      description: "Freiwillige Mathematik-Nachhilfe für einen Oberschüler."
    }
  ],
  education: [
    {
      school: "Hasan Kalyoncu Universität",
      degree: "Computertechnik (Englisch) - Bachelor-Abschluss",
      period: "2019-2025",
      detail: "Vollstipendium"
    },
    {
      school: "Universität Ljubljana",
      degree: "Erasmus+ Austausch",
      period: "2024",
      detail: "Computer Vision Labor"
    },
    {
      school: "Gaziantep Anatolian High School",
      degree: "Oberschule",
      period: "",
      detail: "(TÜRKİYE)"
    }
  ],
  skills: {
    technical: SKILLS_DATA_EN[0].skills.concat(SKILLS_DATA_EN[1].skills, SKILLS_DATA_EN[2].skills, SKILLS_DATA_EN[3].skills),
    soft: [
      "Teamarbeit", "Integrität", "Führung", "Analytische Problemlösung"
    ]
  },
  languages: [
    { name: "Englisch", level: "IELTS: B2" },
    { name: "Deutsch", level: "A2" },
    { name: "Türkisch", level: "Muttersprache" }
  ],
  certificates: [
    "Google AI Essentials (Google)",
    "IELTS Zertifikat (Level: B2)",
    "Advanced Data Analytics (Coursera - Google)",
    "Miuul Machine Learning",
    "Python für Machine Learning (AI Business School)",
    "Google Projektmanagement (Coursera - Google)",
    "Data Analysis Bootcamp (Global AI Hub)",
    "E-Commerce Masterclass (The Dot Academy)"
  ],
  hobbies: [
    "Yoga", "Klavier spielen", "Amateurtheater", "Schach", "Zeichnen"
  ],
  references: RESUME_DATA_EN.references
};

const SKILLS_DATA_DE_RESUME = {
    technical: SKILLS_DATA_EN.flatMap(s => s.skills),
    soft: RESUME_DATA_DE.skills.soft
};
RESUME_DATA_DE.skills = SKILLS_DATA_DE_RESUME as any;


/* ---------------- SPANISH DATA (ES) ---------------- */

const PERSONAL_INFO_ES = {
  ...PERSONAL_INFO_EN,
  title: "CIENTÍFICA DE DATOS / ING. DE IA Y COMPUTACIÓN",
  about: `Me gradué en enero de 2025 de la Universidad Hasan Kalyoncu con una licenciatura en Ingeniería Informática (en inglés, beca completa). A lo largo de mi trayectoria académica, adquirí una amplia experiencia práctica a través de pasantías nacionales e internacionales, proyectos de investigación y colaboraciones académico-industriales, especialmente en los campos de ciencia de datos, inteligencia artificial y desarrollo de software.

Anteriormente, en el verano de 2023, completé una pasantía a largo plazo en una de las empresas corporativas más grandes de la región de Anatolia Sudoriental, donde trabajé activamente en C#, .NET, C# Windows Forms y sistemas ERP como SAP/ABAP. Durante esta pasantía, participé en proyectos listos para producción, contribuí directamente al desarrollo de código y colaboré con equipos multifuncionales en entornos empresariales reales. En mi último año, obtuve una beca Erasmus+ y completé una pasantía internacional de 4 meses en la Universidad de Liubliana, Facultad de Ciencias de la Computación e Informática, donde trabajé en un Laboratorio de Visión por Computadora. En este rol, contribuí a la investigación de visión por computadora, preparación de conjuntos de datos, experimentación de modelos y varios pipelines de CV basados en Python, fortaleciendo mi experiencia en aprendizaje profundo y tecnologías de procesamiento de imágenes.`,
};

const TYPEWRITER_TEXTS_ES = [
  "Soy Ingeniera de IA",
  "Soy Ingeniera Informática",
  "Soy Ingeniera de Machine Learning",
  "Soy Ingeniera de Software"
];

const SKILLS_DATA_ES = SKILLS_DATA_EN;

const PROJECT_CATEGORIES_ES: ProjectCategory[] = JSON.parse(JSON.stringify(PROJECT_CATEGORIES_EN));
PROJECT_CATEGORIES_ES.forEach(cat => {
    if(cat.id === "data-analyze-ml") {
        cat.title = "Análisis de Datos & ML";
        cat.description = "Análisis de datos en profundidad, modelado predictivo y soluciones de aprendizaje automático.";
    } else if (cat.id === "python-projects") {
        cat.title = "Proyectos Python";
        cat.description = "Aplicaciones Python versátiles que van desde la automatización hasta la ciencia de datos.";
    } else if (cat.id === "computer-vision") {
        cat.title = "Visión por Computadora";
        cat.description = "Procesamiento de imágenes avanzado, aprendizaje profundo y sistemas de visión en tiempo real.";
    } else if (cat.id === "ai-projects") {
        cat.title = "Proyectos de IA";
        cat.description = "Aplicaciones de inteligencia artificial de vanguardia y sistemas automatizados.";
    } else if (cat.id === "end-to-end") {
        cat.title = "Proyectos End-to-End";
        cat.description = "Proyectos full-stack y de ciclo completo desde la concepción hasta el despliegue.";
    } else if (cat.id === "other-projects") {
        cat.title = "Otros Proyectos";
        cat.description = "Varios otros proyectos técnicos y experimentos.";
    }
});

const RESUME_DATA_ES = {
  experience: [
    {
      role: "Miembro del Equipo Central",
      company: "Comunidad Kaggle Türkiye",
      period: "Junio 2025 - Presente",
      description: "Organización de eventos de IA y ciencia de datos en Turquía; gestión de patrocinios y asociaciones."
    },
    {
      role: "Becaria de Ciencia de Datos",
      company: "Google AI & Technology Academy",
      period: "Noviembre 2024 - Presente",
      description: "Adquirí valiosa experiencia en gestión de proyectos participando en hackatones, capacitación empresarial y competencias de ciencia de datos."
    },
    {
      role: "Colaboradora - Entrenadora (Tiempo Parcial)",
      company: "Outlier",
      period: "Noviembre 2024 - Presente",
      description: "Trabajé a tiempo parcial entrenando modelos de IA mediante la optimización y experimentación con modelos de lenguaje."
    },
    {
      role: "Voluntaria",
      company: "Fundación T3",
      period: "Abril 2024 - Presente",
      description: "Voluntariado en iniciativas de Aprendizaje Automático, Visión por Computadora y Aprendizaje Profundo."
    },
    {
      role: "Pasante de Intercambio Erasmus+",
      company: "Laboratorio de Visión por Computadora, Univ. de Liubliana",
      period: "Febrero 2024 - Agosto 2024",
      description: "Realicé investigaciones en el campo del aprendizaje automático, visión por computadora y aprendizaje profundo."
    },
    {
      role: "Ingeniera Pasante",
      company: "SANKO Holding A.Ş (SANShine)",
      period: "Julio 2023 - Octubre 2023",
      description: "Asistí en la integración y gestión de sistemas ERP, incluyendo trabajo con SAP, ABAP, SQL, C# y .NET."
    },
    {
      role: "Organizadora de Eventos",
      company: "Miembro del Equipo Central HKU GDSC",
      period: "Septiembre 2023 - Mayo 2024",
      description: "Organicé eventos tecnológicos, gestioné patrocinios y facilité asociaciones para mejorar la participación comunitaria."
    },
    {
      role: "Coordinadora del Equipo de Proyecto",
      company: "2209 TUBITAK",
      period: "Octubre 2023 - Abril 2024",
      description: "Programa de Apoyo a Proyectos de Investigación de Estudiantes Universitarios - Proyecto de Simulación VR."
    },
    {
      role: "Voluntaria de Entrenamiento Individual",
      company: "Plataforma de Entrenamiento Bamboo",
      period: "Octubre 2022 - Diciembre 2024",
      description: "Proporcioné tutoría voluntaria de matemáticas a un estudiante de secundaria."
    }
  ],
  education: [
    {
      school: "Universidad Hasan Kalyoncu",
      degree: "Ingeniería Informática (Inglés) - Licenciatura",
      period: "2019-2025",
      detail: "Beca Completa"
    },
    {
      school: "Universidad de Liubliana",
      degree: "Intercambio Erasmus+",
      period: "2024",
      detail: "Laboratorio de Visión por Computadora"
    },
    {
      school: "Escuela Secundaria de Anatolia Gaziantep",
      degree: "Escuela Secundaria",
      period: "",
      detail: "(TURQUÍA)"
    }
  ],
  skills: {
    technical: SKILLS_DATA_EN[0].skills.concat(SKILLS_DATA_EN[1].skills, SKILLS_DATA_EN[2].skills, SKILLS_DATA_EN[3].skills),
    soft: [
      "Trabajo en Equipo", "Integridad", "Liderazgo", "Resolución Analítica de Problemas"
    ]
  },
  languages: [
    { name: "Inglés", level: "IELTS: B2" },
    { name: "Alemán", level: "A2" },
    { name: "Turco", level: "Nativo" }
  ],
  certificates: [
    "Google AI Essentials (Google)",
    "Certificado IELTS (Nivel: B2)",
    "Análisis de Datos Avanzado (Coursera - Google)",
    "Aprendizaje Automático Miuul",
    "Python para Aprendizaje Automático (AI Business School)",
    "Gestión de Proyectos de Google (Coursera - Google)",
    "Bootcamp de Análisis de Datos (Global AI Hub)",
    "Clase Magistral de Comercio Electrónico (The Dot Academy)"
  ],
  hobbies: [
    "Hacer Yoga", "Tocar el Piano", "Actuación Teatral Amateur", "Jugar Ajedrez", "Dibujar"
  ],
  references: RESUME_DATA_EN.references
};

const SKILLS_DATA_ES_RESUME = {
    technical: SKILLS_DATA_EN.flatMap(s => s.skills),
    soft: RESUME_DATA_ES.skills.soft
};
RESUME_DATA_ES.skills = SKILLS_DATA_ES_RESUME as any;


/* ---------------- UI TEXTS ---------------- */

const UI_LABELS = {
  en: {
    nav: { home: 'Home', resume: 'Resume / CV', projects: 'Projects', contact: 'Contact' },
    hero: {
      welcome: 'Welcome to my portfolio',
      hi: "Hi, It's",
      iam: "I am",
      desc_part1: "Passionate about transforming complex data into actionable insights. With expertise in",
      desc_highlight1: "Machine Learning",
      desc_part2: ",",
      desc_highlight2: "AI systems",
      desc_part3: ", and",
      desc_highlight3: "Data Science",
      desc_part4: ", I build intelligent solutions that drive innovation.",
      viewProjects: "View Projects",
      downloadCv: "Download CV"
    },
    about: {
      title: "About",
      me: "Me",
      certificates: "Certificates"
    },
    skills: {
      title: "Technical",
      highlight: "Skills",
      subtitle: "My technical expertise and toolkit"
    },
    projects: {
      title: "My",
      highlight: "Projects",
      desc: "Explore my technical portfolio across various domains. Click on a category to view detailed projects.",
      viewCategory: "View Category"
    },
    contact: {
      tag: "Contact Me",
      title: "Let's Work",
      highlight: "Together",
      desc: "I'm currently open for new opportunities and collaborations in AI, Data Science, and Software Engineering. Have a project in mind or just want to say hi? I'd love to hear from you.",
      emailLabel: "Email Me",
      locationLabel: "Location",
      connectLabel: "Connect with me",
      formTitle: "Send me a message",
      nameLabel: "Your Name",
      emailInputLabel: "Your Email",
      messageLabel: "Message",
      sendButton: "Send Message",
      rights: "All Rights Reserved."
    },
    resume: {
      myProjects: "My Projects",
      aboutMe: "About Me",
      workExp: "Work Experience",
      education: "Education",
      languages: "Languages",
      certificates: "Certificates",
      technicalSkills: "Technical Skills",
      softSkills: "Soft Skills",
      references: "References",
      hobbies: "Hobbies"
    },
    projectList: {
      back: "Back",
      liveDemo: "Live Demo",
      code: "Code",
      noProjects: "No projects found",
      tryFilters: "Try adjusting the filters."
    },
    modal: {
      liveDemo: "Live Demo",
      sourceCode: "Source Code",
      comingSoon: "Projects Coming Soon...",
      workingOn: "I'm currently working on some exciting things in this domain."
    }
  },
  tr: {
    nav: { home: 'Anasayfa', resume: 'Özgeçmiş', projects: 'Projeler', contact: 'İletişim' },
    hero: {
      welcome: 'Portföyüme hoş geldiniz',
      hi: "Merhaba, Ben",
      iam: "Ben",
      desc_part1: "Karmaşık verileri eyleme dönüştürülebilir içgörülere çevirme konusunda tutkuluyum.",
      desc_highlight1: "Makine Öğrenimi",
      desc_part2: ",",
      desc_highlight2: "Yapay Zeka",
      desc_part3: " ve",
      desc_highlight3: "Veri Bilimi",
      desc_part4: " alanlarındaki uzmanlığımla, inovasyonu yönlendiren akıllı çözümler geliştiriyorum.",
      viewProjects: "Projeleri Gör",
      downloadCv: "CV İndir"
    },
    about: {
      title: "Hakkımda",
      me: "", 
      certificates: "Sertifikalar"
    },
    skills: {
      title: "Teknik",
      highlight: "Yetenekler",
      subtitle: "Teknik uzmanlığım ve araç setim"
    },
    projects: {
      title: "Projelerim",
      highlight: "",
      desc: "Çeşitli alanlardaki teknik portföyümü keşfedin. Detaylı projeleri görmek için bir kategoriye tıklayın.",
      viewCategory: "Kategoriyi İncele"
    },
    contact: {
      tag: "Bana Ulaşın",
      title: "Birlikte",
      highlight: "Çalışalım",
      desc: "Şu anda Yapay Zeka, Veri Bilimi ve Yazılım Mühendisliği alanlarında yeni fırsatlara ve iş birliklerine açığım. Aklınızda bir proje mi var ya da sadece merhaba demek mi istiyorsunuz? Sizi duymaktan memnuniyet duyarım.",
      emailLabel: "Bana E-posta Gönder",
      locationLabel: "Konum",
      connectLabel: "Benimle Bağlantı Kurun",
      formTitle: "Bana mesaj gönder",
      nameLabel: "Adınız",
      emailInputLabel: "E-postanız",
      messageLabel: "Mesajınız",
      sendButton: "Mesaj Gönder",
      rights: "Tüm Hakları Saklıdır."
    },
    resume: {
      myProjects: "Projelerim",
      aboutMe: "Hakkımda",
      workExp: "İş Deneyimi",
      education: "Eğitim",
      languages: "Diller",
      certificates: "Sertifikalar",
      technicalSkills: "Teknik Yetenekler",
      softSkills: "Yetenekler",
      references: "Referanslar",
      hobbies: "Hobiler"
    },
    projectList: {
      back: "Geri",
      liveDemo: "Canlı Demo",
      code: "Kod",
      noProjects: "Proje bulunamadı",
      tryFilters: "Filtreleri değiştirmeyi deneyin."
    },
    modal: {
      liveDemo: "Canlı Demo",
      sourceCode: "Kaynak Kodu",
      comingSoon: "Projeler Yakında...",
      workingOn: "Şu anda bu alanda heyecan verici şeyler üzerinde çalışıyorum."
    }
  },
  de: {
    nav: { home: 'Startseite', resume: 'Lebenslauf / CV', projects: 'Projekte', contact: 'Kontakt' },
    hero: {
      welcome: 'Willkommen in meinem Portfolio',
      hi: "Hallo, Ich bin",
      iam: "Ich bin",
      desc_part1: "Leidenschaftlich darin, komplexe Daten in umsetzbare Erkenntnisse zu verwandeln. Mit Expertise in",
      desc_highlight1: "Maschinellem Lernen",
      desc_part2: ",",
      desc_highlight2: "KI-Systemen",
      desc_part3: " und",
      desc_highlight3: "Data Science",
      desc_part4: " baue ich intelligente Lösungen, die Innovation vorantreiben.",
      viewProjects: "Projekte ansehen",
      downloadCv: "CV herunterladen"
    },
    about: {
      title: "Über",
      me: "Mich",
      certificates: "Zertifikate"
    },
    skills: {
      title: "Technische",
      highlight: "Fähigkeiten",
      subtitle: "Meine technische Expertise und Werkzeuge"
    },
    projects: {
      title: "Meine",
      highlight: "Projekte",
      desc: "Entdecken Sie mein technisches Portfolio in verschiedenen Bereichen. Klicken Sie auf eine Kategorie, um detaillierte Projekte anzuzeigen.",
      viewCategory: "Kategorie ansehen"
    },
    contact: {
      tag: "Kontaktieren Sie mich",
      title: "Lassen Sie uns",
      highlight: "zusammenarbeiten",
      desc: "Ich bin derzeit offen für neue Möglichkeiten und Kooperationen in den Bereichen KI, Data Science und Software Engineering. Haben Sie ein Projekt im Sinn oder möchten Sie einfach nur Hallo sagen? Ich würde mich freuen, von Ihnen zu hören.",
      emailLabel: "Schreiben Sie mir",
      locationLabel: "Standort",
      connectLabel: "Vernetzen Sie sich mit mir",
      formTitle: "Senden Sie mir eine Nachricht",
      nameLabel: "Ihr Name",
      emailInputLabel: "Ihre E-Mail",
      messageLabel: "Nachricht",
      sendButton: "Nachricht senden",
      rights: "Alle Rechte vorbehalten."
    },
    resume: {
      myProjects: "Meine Projekte",
      aboutMe: "Über mich",
      workExp: "Berufserfahrung",
      education: "Ausbildung",
      languages: "Sprachen",
      certificates: "Zertifikate",
      technicalSkills: "Technische Fähigkeiten",
      softSkills: "Soft Skills",
      references: "Referenzen",
      hobbies: "Hobbys"
    },
    projectList: {
      back: "Zurück",
      liveDemo: "Live-Demo",
      code: "Code",
      noProjects: "Keine Projekte gefunden",
      tryFilters: "Versuchen Sie, die Filter anzupassen."
    },
    modal: {
      liveDemo: "Live-Demo",
      sourceCode: "Quellcode",
      comingSoon: "Projekte folgen bald...",
      workingOn: "Ich arbeite derzeit an einigen spannenden Dingen in diesem Bereich."
    }
  },
  es: {
    nav: { home: 'Inicio', resume: 'Currículum / CV', projects: 'Proyectos', contact: 'Contacto' },
    hero: {
      welcome: 'Bienvenido a mi portafolio',
      hi: "Hola, Soy",
      iam: "Soy",
      desc_part1: "Apasionada por transformar datos complejos en conocimientos prácticos. Con experiencia en",
      desc_highlight1: "Aprendizaje Automático",
      desc_part2: ",",
      desc_highlight2: "sistemas de IA",
      desc_part3: " y",
      desc_highlight3: "Ciencia de Datos",
      desc_part4: ", construyo soluciones inteligentes que impulsan la innovación.",
      viewProjects: "Ver Proyectos",
      downloadCv: "Descargar CV"
    },
    about: {
      title: "Sobre",
      me: "Mí",
      certificates: "Certificados"
    },
    skills: {
      title: "Habilidades",
      highlight: "Técnicas",
      subtitle: "Mi experiencia técnica y herramientas"
    },
    projects: {
      title: "Mis",
      highlight: "Proyectos",
      desc: "Explore mi portafolio técnico en varios dominios. Haga clic en una categoría para ver proyectos detallados.",
      viewCategory: "Ver Categoría"
    },
    contact: {
      tag: "Contáctame",
      title: "Trabajemos",
      highlight: "Juntos",
      desc: "Actualmente estoy abierta a nuevas oportunidades y colaboraciones en IA, Ciencia de Datos e Ingeniería de Software. ¿Tienes un proyecto en mente o simplemente quieres saludar? Me encantaría saber de ti.",
      emailLabel: "Envíame un correo",
      locationLabel: "Ubicación",
      connectLabel: "Conéctate conmigo",
      formTitle: "Envíame un mensaje",
      nameLabel: "Tu Nombre",
      emailInputLabel: "Tu Correo",
      messageLabel: "Mensaje",
      sendButton: "Enviar Mensaje",
      rights: "Todos los derechos reservados."
    },
    resume: {
      myProjects: "Mis Proyectos",
      aboutMe: "Sobre Mí",
      workExp: "Experiencia Laboral",
      education: "Educación",
      languages: "Idiomas",
      certificates: "Certificados",
      technicalSkills: "Habilidades Técnicas",
      softSkills: "Habilidades Blandas",
      references: "Referencias",
      hobbies: "Pasatiempos"
    },
    projectList: {
      back: "Atrás",
      liveDemo: "Demo en Vivo",
      code: "Código",
      noProjects: "No se encontraron proyectos",
      tryFilters: "Intenta ajustar los filtros."
    },
    modal: {
      liveDemo: "Demo en Vivo",
      sourceCode: "Código Fuente",
      comingSoon: "Proyectos Próximamente...",
      workingOn: "Actualmente estoy trabajando en algunas cosas emocionantes en este dominio."
    }
  }
};


/* ---------------- EXPORTS ---------------- */

export const DATA_EN = {
  PERSONAL_INFO: PERSONAL_INFO_EN,
  SOCIAL_LINKS,
  TYPEWRITER_TEXTS: TYPEWRITER_TEXTS_EN,
  SKILLS_DATA: SKILLS_DATA_EN,
  PROJECT_CATEGORIES: PROJECT_CATEGORIES_EN,
  // We recreate the categories mapped array for each lang
  CATEGORIES: PROJECT_CATEGORIES_EN.map(({ id, title, count, emoji, path, gradient }) => ({
    id, title, count, emoji, path, gradient
  })),
  RESUME_DATA: RESUME_DATA_EN,
  UI: UI_LABELS.en
};

export const DATA_TR = {
  PERSONAL_INFO: PERSONAL_INFO_TR,
  SOCIAL_LINKS,
  TYPEWRITER_TEXTS: TYPEWRITER_TEXTS_TR,
  SKILLS_DATA: SKILLS_DATA_TR,
  PROJECT_CATEGORIES: PROJECT_CATEGORIES_TR,
  CATEGORIES: PROJECT_CATEGORIES_TR.map(({ id, title, count, emoji, path, gradient }) => ({
    id, title, count, emoji, path, gradient
  })),
  RESUME_DATA: RESUME_DATA_TR,
  UI: UI_LABELS.tr
};

export const DATA_DE = {
  PERSONAL_INFO: PERSONAL_INFO_DE,
  SOCIAL_LINKS,
  TYPEWRITER_TEXTS: TYPEWRITER_TEXTS_DE,
  SKILLS_DATA: SKILLS_DATA_DE,
  PROJECT_CATEGORIES: PROJECT_CATEGORIES_DE,
  CATEGORIES: PROJECT_CATEGORIES_DE.map(({ id, title, count, emoji, path, gradient }) => ({
    id, title, count, emoji, path, gradient
  })),
  RESUME_DATA: RESUME_DATA_DE,
  UI: UI_LABELS.de
};

export const DATA_ES = {
  PERSONAL_INFO: PERSONAL_INFO_ES,
  SOCIAL_LINKS,
  TYPEWRITER_TEXTS: TYPEWRITER_TEXTS_ES,
  SKILLS_DATA: SKILLS_DATA_ES,
  PROJECT_CATEGORIES: PROJECT_CATEGORIES_ES,
  CATEGORIES: PROJECT_CATEGORIES_ES.map(({ id, title, count, emoji, path, gradient }) => ({
    id, title, count, emoji, path, gradient
  })),
  RESUME_DATA: RESUME_DATA_ES,
  UI: UI_LABELS.es
};

// For backward compatibility (if any files import directly) - defaulting to EN
export const PERSONAL_INFO = PERSONAL_INFO_EN;
export const TYPEWRITER_TEXTS = TYPEWRITER_TEXTS_EN;
export const SKILLS_DATA = SKILLS_DATA_EN;
export const PROJECT_CATEGORIES = PROJECT_CATEGORIES_EN;
export const CATEGORIES = DATA_EN.CATEGORIES;
export const RESUME_DATA = RESUME_DATA_EN;
