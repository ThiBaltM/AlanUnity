
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;
using Unity.VisualScripting;
using UnityEngine.UIElements;

public class AlanScript : Agent
{
    public GameObject head;
    public GameObject leftHeel;
    public GameObject leftTibia;
    public GameObject leftFeet;

    public GameObject rightHeel;
    public GameObject rightTibia;
    public GameObject rightFeet;

    public GameObject objectiv;

    private ConfigurableJoint leftHeelJoint;
    private HingeJoint leftTibiaJoint;
    private HingeJoint leftFeetJoint;

    private ConfigurableJoint rightHeelJoint;
    private HingeJoint rightTibiaJoint;
    private HingeJoint rightFeetJoint;


    //initial position
    // Variables pour sauvegarder les positions et rotations initiales
    private Vector3 headInitialPosition;
    private Quaternion headInitialRotation;
    private Vector3 leftHeelInitialPosition;
    private Quaternion leftHeelInitialRotation;
    private Vector3 leftTibiaInitialPosition;
    private Quaternion leftTibiaInitialRotation;
    private Vector3 leftFeetInitialPosition;
    private Quaternion leftFeetInitialRotation;
    private Vector3 rightHeelInitialPosition;
    private Quaternion rightHeelInitialRotation;
    private Vector3 rightTibiaInitialPosition;
    private Quaternion rightTibiaInitialRotation;
    private Vector3 rightFeetInitialPosition;
    private Quaternion rightFeetInitialRotation;

    private CollisionManager leftFootCollisionManager;
    private CollisionManager rightFootCollisionManager;
    private CollisionManager headCollisionManager;

    // Start is called before the first frame update
    void Start()
    {

        leftFootCollisionManager = leftFeet.GetComponent<CollisionManager>();
        rightFootCollisionManager = rightFeet.GetComponent<CollisionManager>();
        headCollisionManager = head.GetComponent<CollisionManager>();

        leftHeelJoint = leftHeel.GetComponent<ConfigurableJoint>();
        leftTibiaJoint = leftTibia.GetComponent<HingeJoint>();
        leftFeetJoint = leftFeet.GetComponent<HingeJoint>();

        rightHeelJoint = rightHeel.GetComponent<ConfigurableJoint>();
        rightTibiaJoint = rightTibia.GetComponent<HingeJoint>();
        rightFeetJoint = rightFeet.GetComponent<HingeJoint>();

        // Sauvegarder les positions et rotations initiales
        headInitialPosition = head.transform.position;
        headInitialRotation = head.transform.rotation;
        leftHeelInitialPosition = leftHeel.transform.position;
        leftHeelInitialRotation = leftHeel.transform.rotation;
        leftTibiaInitialPosition = leftTibia.transform.position;
        leftTibiaInitialRotation = leftTibia.transform.rotation;
        leftFeetInitialPosition = leftFeet.transform.position;
        leftFeetInitialRotation = leftFeet.transform.rotation;
        rightHeelInitialPosition = rightHeel.transform.position;
        rightHeelInitialRotation = rightHeel.transform.rotation;
        rightTibiaInitialPosition = rightTibia.transform.position;
        rightTibiaInitialRotation = rightTibia.transform.rotation;
        rightFeetInitialPosition = rightFeet.transform.position;
        rightFeetInitialRotation = rightFeet.transform.rotation;
    }


    public override void OnEpisodeBegin()
    {
        // Réinitialiser les positions et rotations des segments
        head.transform.position = headInitialPosition;
        head.transform.rotation = headInitialRotation;
        leftHeel.transform.position = leftHeelInitialPosition;
        leftHeel.transform.rotation = leftHeelInitialRotation;
        leftTibia.transform.position = leftTibiaInitialPosition;
        leftTibia.transform.rotation = leftTibiaInitialRotation;
        leftFeet.transform.position = leftFeetInitialPosition;
        leftFeet.transform.rotation = leftFeetInitialRotation;
        rightHeel.transform.position = rightHeelInitialPosition;
        rightHeel.transform.rotation = rightHeelInitialRotation;
        rightTibia.transform.position = rightTibiaInitialPosition;
        rightTibia.transform.rotation = rightTibiaInitialRotation;
        rightFeet.transform.position = rightFeetInitialPosition;
        rightFeet.transform.rotation = rightFeetInitialRotation;

        // Réinitialiser les angles des articulations
        ResetHeelAngles(leftHeelJoint);
        ResetJointAngles(leftTibiaJoint);
        ResetJointAngles(leftFeetJoint);
        ResetHeelAngles(rightHeelJoint);
        ResetJointAngles(rightTibiaJoint);
        ResetJointAngles(rightFeetJoint);
    }

    void ResetJointAngles(HingeJoint joint)
    {
        JointSpring spring = joint.spring;
        spring.targetPosition = 0; // Position initiale
        joint.spring = spring;
    }

    public void ResetHeelAngles(ConfigurableJoint joint)
    {
        // Créer un quaternion pour représenter l'angle de rotation cible
        Quaternion targetRotation = Quaternion.Euler(0, 0, 0); // Position initiale

        // Appliquer la rotation cible au joint configurable
        joint.targetRotation = targetRotation;

        // Définir les paramètres du JointDrive pour les mouvements angulaires
        JointDrive angularDrive = new JointDrive
        {
            positionSpring = 1000,
            positionDamper = 10,
            maximumForce = 1000
        };

        joint.angularXDrive = angularDrive;
        joint.angularYZDrive = angularDrive;
    }


    public override void CollectObservations(VectorSensor sensor)
    {
        // Ajouter la position de la tête (3 composantes pour le vecteur)
        sensor.AddObservation(head.transform.position);

        // Ajouter les angles de la tête sur les trois axes
        Vector3 headAngles = head.transform.eulerAngles;
        sensor.AddObservation(headAngles.x);
        sensor.AddObservation(headAngles.y);
        sensor.AddObservation(headAngles.z);

        // Récupérer l'orientation locale des articulations de la hanche
        Vector3 leftHipJointAngles = leftHeelJoint.transform.localRotation.eulerAngles;
        Vector3 rightHipJointAngles = rightHeelJoint.transform.localRotation.eulerAngles;

        // Ajouter les angles X et Z des hanches (éviter Y si ce n'est pas utile)
        sensor.AddObservation(leftHipJointAngles.x);
        sensor.AddObservation(leftHipJointAngles.z);

        sensor.AddObservation(rightHipJointAngles.x);
        sensor.AddObservation(rightHipJointAngles.z);

        // Ajouter les angles des articulations pour la jambe gauche
        sensor.AddObservation(leftTibiaJoint.angle);
        sensor.AddObservation(leftFeetJoint.angle);

        // Ajouter les angles des articulations pour la jambe droite
        sensor.AddObservation(rightTibiaJoint.angle);
        sensor.AddObservation(rightFeetJoint.angle);

        // Ajouter des booléens pour indiquer si les pieds touchent le sol
        sensor.AddObservation(leftFootCollisionManager.GetIsGrounded() ? 1.0f : 0.0f);
        sensor.AddObservation(rightFootCollisionManager.GetIsGrounded() ? 1.0f : 0.0f);

        // Vecteur direction du personnage
        Vector3 forwardDirection = head.transform.forward;

        // Vecteur direction vers la cible
        Vector3 targetDirection = (objectiv.transform.position - head.transform.position).normalized;

        // Calculer l'angle entre les deux vecteurs
        float angle = Vector3.SignedAngle(forwardDirection, targetDirection, Vector3.up);

        // Ajouter l'angle comme observation
        sensor.AddObservation(angle / 180.0f); // Normalisation (-1 à 1)
        Debug.Log(angle / 180.0f);
    }


    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (actionBuffers.ContinuousActions.Length < 8)
        {
            Debug.LogError($"Trop peu d'actions reçues ! Attendu : 8, Reçu : {actionBuffers.ContinuousActions.Length}");
            return;
        }

        // Récupérer les actions envoyées par Python
        float leftHipXTarget = actionBuffers.ContinuousActions[0];  // Hanche gauche (avant/arrière)
        float leftHipZTarget = actionBuffers.ContinuousActions[1];  // Hanche gauche (côtés)
        float leftKneeTarget = actionBuffers.ContinuousActions[2];  // Genou gauche
        float leftAnkleTarget = actionBuffers.ContinuousActions[3]; // Cheville gauche

        float rightHipXTarget = actionBuffers.ContinuousActions[4];  // Hanche droite (avant/arrière)
        float rightHipZTarget = actionBuffers.ContinuousActions[5];  // Hanche droite (côtés)
        float rightKneeTarget = actionBuffers.ContinuousActions[6];  // Genou droit
        float rightAnkleTarget = actionBuffers.ContinuousActions[7]; // Cheville droite

        // Appliquer les actions aux articulations
        SetHipAngles(leftHeelJoint, leftHipXTarget, leftHipZTarget);
        SetHipAngles(rightHeelJoint, rightHipXTarget, rightHipZTarget);

        SetJointTarget(leftTibiaJoint, leftKneeTarget);
        SetJointTarget(leftFeetJoint, leftAnkleTarget);
        SetJointTarget(rightTibiaJoint, rightKneeTarget);
        SetJointTarget(rightFeetJoint, rightAnkleTarget);
    }



    public void SetJointTarget(HingeJoint joint, float angle)
    {
        if (angle < -1)
        {
            angle = -1;
        }else if (angle > 1)
        {
            angle = 1;
        }
        // Définir les angles minimaux et maximaux pour le HingeJoint
        float minAngle = 0;
        float maxAngle = 90;

        // Calculer l'angle proportionnel
        float targetAngle = Mathf.Lerp(minAngle, maxAngle, (angle + 1) / 2f);

        // 1. Créer un objet de type JointSpring
        JointSpring spring = joint.spring;

        // 2. Définir la force et l'amortissement
        spring.spring = 1000f;
        spring.damper = 1000f; // Amortissement pour éviter les oscillations

        // Définir la position cible
        spring.targetPosition = targetAngle;

        joint.spring = spring;
    }

    public void SetHipAngles(ConfigurableJoint heel, float angleX, float angleZ)
    {
        if (angleX < -1)
        {
            angleX = -1;
        }
        else if (angleX > 1)
        {
            angleX = 1;
        }
        if (angleZ < -1)
        {
            angleZ = -1;
        }
        else if (angleZ > 1)
        {
            angleZ = 1;
        }
        // Calculer les angles proportionnels
        float targetAngleX = Mathf.Lerp(-45, 45, (angleX + 1) / 2f);
        float targetAngleZ = Mathf.Lerp(-90, 90, (angleZ + 1) / 2f);

        // Créer un quaternion pour représenter l'angle de rotation cible en fonction des angles calculés
        Quaternion targetRotation = Quaternion.Euler(targetAngleX, 0, targetAngleZ);

        // Appliquer la rotation cible au joint configurable
        heel.targetRotation = targetRotation;
    }

    void getObjectivDistance()
    {   
        double distanceX = objectiv.transform.position.x - head.transform.position.x;
        double distanceY = objectiv.transform.position.y - head.transform.position.y;
        float distance = (float)Math.Sqrt(distanceX * distanceX + distanceY * distanceY);
        AddReward(distance*(-10));
    }

    void Update()
    {
        // Demander des décisions et actions à chaque frame
        RequestDecision();
        RequestAction();
        AddReward(0.1f);

        if (headCollisionManager.GetIsGrounded())
        {
            getObjectivDistance();
            EndEpisode();
        }
    }

}



