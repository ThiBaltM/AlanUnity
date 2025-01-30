
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;

public class AlanScript : Agent
{
    public GameObject head;
    public GameObject leftHeel;
    public GameObject leftTibia;
    public GameObject leftFeet;

    public GameObject rightHeel;
    public GameObject rightTibia;
    public GameObject rightFeet;

    private HingeJoint leftHeelJoint;
    private HingeJoint leftTibiaJoint;
    private HingeJoint leftFeetJoint;

    private HingeJoint rightHeelJoint;
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

    // Start is called before the first frame update
    void Start()
    {

        leftFootCollisionManager = leftFeet.GetComponent<CollisionManager>();
        rightFootCollisionManager = rightFeet.GetComponent<CollisionManager>();

        leftHeelJoint = leftHeel.GetComponent<HingeJoint>();
        leftTibiaJoint = leftTibia.GetComponent<HingeJoint>();
        leftFeetJoint = leftFeet.GetComponent<HingeJoint>();

        rightHeelJoint = rightHeel.GetComponent<HingeJoint>();
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
        ResetJointAngles(leftHeelJoint);
        ResetJointAngles(leftTibiaJoint);
        ResetJointAngles(leftFeetJoint);
        ResetJointAngles(rightHeelJoint);
        ResetJointAngles(rightTibiaJoint);
        ResetJointAngles(rightFeetJoint);
    }

    void ResetJointAngles(HingeJoint joint)
    {
        JointSpring spring = joint.spring;
        spring.targetPosition = 0; // Position initiale
        joint.spring = spring;
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
        Debug.Log(leftFootCollisionManager.GetIsGrounded());
    }


    public override void OnActionReceived(ActionBuffers actionBuffers){
        // Appliquer les actions de l'IA aux joints
        float leftHeelTarget = actionBuffers.ContinuousActions[0];
        float leftTibiaTarget = actionBuffers.ContinuousActions[1];
        float leftFeetTarget = actionBuffers.ContinuousActions[2];

        float rightHeelTarget = actionBuffers.ContinuousActions[0];
        float rightTibiaTarget = actionBuffers.ContinuousActions[1];
        float rightFeetTarget = actionBuffers.ContinuousActions[2];

        SetJointTarget(leftHeelJoint, leftHeelTarget);
        SetJointTarget(leftTibiaJoint, leftTibiaTarget);
        SetJointTarget(leftFeetJoint, leftFeetTarget);

        SetJointTarget(rightHeelJoint, rightHeelTarget);
        SetJointTarget(rightTibiaJoint, rightTibiaTarget);
        SetJointTarget(rightFeetJoint, rightFeetTarget);
    }

    void SetJointTarget(HingeJoint joint, float target)    {
            JointSpring spring = joint.spring;
            spring.targetPosition = target;
            joint.spring = spring;
    }

    void Update()
    {
        // Demander des décisions et actions à chaque frame
        RequestDecision();
        RequestAction();
    }

}



